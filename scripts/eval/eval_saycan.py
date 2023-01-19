"""
Usage:

PYTHONPATH=. python scripts/eval/eval_saycan.py
--planner-config configs/pybullet/planners/ablation/policy_cem.yaml 
--env-config configs/pybullet/envs/t2m/official/tasks/task0.yaml
--policy-checkpoints
    models/20230106/complete_q_multistage/pick_0/ckpt_model_1000000.pt 
    models/20230101/complete_q_multistage/place_0/best_model.pt 
    models/20230101/complete_q_multistage/pull_0/best_model.pt
    models/20230101/complete_q_multistage/push_0/best_model.pt
--dynamics-checkpoint models/official/select_model/dynamics/best_model.pt
--pddl-domain configs/pybullet/envs/t2m/official/tasks/symbolic_domain.pddl
--pddl-problem configs/pybullet/envs/t2m/official/tasks/task0_symbolic.pddl
--verbose 0 --engine davinci --gui 1
--path plots/20230117/eval_saycan
"""

import functools
import random
from temporal_policies import agents, envs, planners
from temporal_policies.dynamics.base import Dynamics
from temporal_policies.envs.pybullet.table.objects import Object

import tqdm
import tabulate

from temporal_policies.planners.utils import get_printable_object_relationships_str

tabulate.PRESERVE_WHITESPACE = True
from tabulate import tabulate
from symbolic import _and, parse_proposition

from termcolor import colored
import argparse
import pathlib
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import json


import torch
import numpy as np
from helm.common.authentication import Authentication
from configs.base_config import LMConfig

from temporal_policies.envs.pybullet.table import predicates
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.evaluation.utils import (
    get_callable_goal_props,
    get_object_relationships,
    get_possible_props,
    get_prop_testing_objs,
    is_satisfy_goal_props,
    seed_generator,
)
from temporal_policies.task_planners.goals import get_goal_from_lm, is_valid_goal_props

from temporal_policies.task_planners.lm_data_structures import APIType
from temporal_policies.task_planners.lm_utils import (
    authenticate,
    get_examples_from_json_dir,
    load_lm_cache,
    save_lm_cache,
)
from temporal_policies.envs.pybullet.table import (
    predicates,
    primitives as table_primitives,
)
from temporal_policies.task_planners.task_plans import (
    get_action_scores_from_lm,
    get_next_actions_from_lm,
)
from temporal_policies.utils import tensors


def get_values(
    observation: np.ndarray,
    actions: np.ndarray,
    primitives: List[table_primitives.Primitive],
    policies: Sequence[agents.Agent],
    dynamics: Dynamics,
    device: torch.device,
) -> List[float]:
    """Get the value of the observation for each primitive."""
    value_fns = [policies[primitive.idx_policy].critic for primitive in primitives]
    policy_fns = [policies[primitive.idx_policy] for primitive in primitives]
    # set to eval mode
    for policy_fn in policy_fns:
        policy_fn.eval_mode()

    decode_fns = [
        functools.partial(dynamics.decode, primitive=primitive)
        for primitive in primitives
    ]
    values: List = []
    with torch.no_grad():
        t_observation = tensors.from_numpy(observation, device=device)
        for i in range(len(policy_fns)):
            try:
                policy_state = decode_fns[i](t_observation)
                q_s_a = value_fns[i].predict(
                    policy_state, tensors.from_numpy(actions[i], device=device)
                )
            except RuntimeError as e:
                q_s_a = torch.tensor(0, dtype=torch.float32)
                print(f"Failed to predict Q for {primitives[i]}, returning Q = 0.")
            values.append(q_s_a.item())
    return values


def get_policy_actions(
    observation: np.ndarray,
    primitives: List[table_primitives.Primitive],
    policies: Sequence[agents.Agent],
    dynamics: Dynamics,
    device: torch.device,
) -> List[int]:
    """Get the action of the policy for each primitive and the current observation."""
    policy_fns = [policies[primitive.idx_policy] for primitive in primitives]

    # set to eval mode
    for policy_fn in policy_fns:
        policy_fn.eval_mode()

    decode_fns = [
        functools.partial(dynamics.decode, primitive=primitive)
        for primitive in primitives
    ]

    actions: List = []
    with torch.no_grad():
        t_observation = tensors.from_numpy(observation, device=device)
        for i in range(len(policy_fns)):
            try:
                policy_state = decode_fns[i](t_observation)
                action = policy_fns[i].actor.predict(policy_state)
            except RuntimeError as e:
                action = torch.tensor(
                    policy_fns[i].action_space.sample(), dtype=torch.float32
                )
                print(
                    f"Failed to predict action for {primitives[i]}, using random action instead."
                )
            actions.append(action.cpu().numpy())
    return actions


def format_saycan_scoring_table(
    table_headers: List[str],
    potential_actions_str: List[str],
    lm_action_scores: List[float],
    value_action_scores: List[float],
    overall_scores: List[float],
    action_pad_width: int = 30,
) -> str:
    # sort the potential actions by overall score and sort the scores accordingly
    potential_actions_str = [
        x for _, x in sorted(zip(overall_scores, potential_actions_str), reverse=True)
    ]
    # pad the actions to be the same length
    potential_actions_str = [
        action.ljust(action_pad_width) for action in potential_actions_str
    ]
    rounded_lm_action_scores = [
        np.round(lm_action_score, 2) for lm_action_score in lm_action_scores
    ]
    rounded_value_action_scores = [
        np.round(value_action_score, 2) for value_action_score in value_action_scores
    ]
    rounded_overall_scores = [
        np.round(overall_score, 2) for overall_score in overall_scores
    ]
    rounded_lm_action_scores = [
        x
        for _, x in sorted(zip(overall_scores, rounded_lm_action_scores), reverse=True)
    ]
    rounded_value_action_scores = [
        x
        for _, x in sorted(
            zip(overall_scores, rounded_value_action_scores), reverse=True
        )
    ]
    rounded_overall_scores = [
        x for _, x in sorted(zip(overall_scores, rounded_overall_scores), reverse=True)
    ]
    formatted_table_str = tabulate(
        zip(
            potential_actions_str,
            rounded_lm_action_scores,
            rounded_value_action_scores,
            rounded_overall_scores,
        ),
        headers=table_headers,
        tablefmt="plain",
    )
    return formatted_table_str


def eval_saycan(
    planner_config: Union[str, pathlib.Path],
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    closed_loop: int,
    pddl_domain: str,  # file path
    pddl_root_dir: str,
    pddl_domain_name: str,
    pddl_problem: str,  # file path
    max_depth: int = 10,
    timeout: float = 10.0,
    verbose: bool = False,
    load_path: Optional[Union[str, pathlib.Path]] = None,
    seed: Optional[int] = 0,
    gui: Optional[int] = None,
    lm_cache_file: Optional[Union[str, pathlib.Path]] = None,
    n_examples: Optional[int] = 1,
    custom_in_context_example_robot_prompt: str = "Top 1 robot action sequences: ",
    custom_in_context_example_robot_format: str = "python_list_of_lists",
    custom_robot_prompt: str = "Top 2 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_done"
    ] = "python_list",
    engine: Optional[str] = None,
    temperature: Optional[int] = 0,
    logprobs: Optional[int] = 1,
    api_type: Optional[APIType] = APIType.HELM,
    max_tokens: Optional[int] = 100,
    auth: Optional[Authentication] = None,
    use_ground_truth_goal_props: bool = False,
):
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # these would perhaps belong in .../prompts/
    examples = get_examples_from_json_dir(
        "configs/pybullet/envs/t2m/official/prompts/hook_reach/"
    )

    for example in examples:
        example.custom_robot_action_sequence_format = (
            custom_robot_action_sequence_format
        )

    examples = random.sample(examples, n_examples)
    lm_cfg: LMConfig = LMConfig(
        engine=engine,
        temperature=temperature,
        logprobs=logprobs,
        echo=False,
        api_type=api_type,
        max_tokens=max_tokens,
    )
    lm_cache: Dict[str, str] = load_lm_cache(pathlib.Path(lm_cache_file))

    # Load environment.
    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)
    # Load planner.
    planner = planners.load(
        config=planner_config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        scod_checkpoints=scod_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )

    path = pathlib.Path(path) / env.name

    prop_testing_objs: Dict[str, Object] = get_prop_testing_objs(env)
    available_predicates: List[str] = [
        parse_proposition(prop)[0] for prop in env.supported_predicates
    ]
    possible_props: List[predicates.Predicate] = get_possible_props(
        env.objects, available_predicates
    )

    num_successes_on_predicted_goal_props: int = 0
    num_successes_on_ground_truth_goal_props: int = 0
    pbar = tqdm.tqdm(
        seed_generator(num_eval, load_path), f"Evaluate {path.name}", dynamic_ncols=True
    )
    for idx_iter, (seed, loaded_plan) in enumerate(pbar):
        goal_props_predicted: List[str]
        objects: List[str]
        object_relationships: List[str]
        all_prior_object_relationships: List[List[str]] = []
        all_executed_actions: List[str] = []

        observation, info = env.reset()

        # get goal props
        objects = list(env.objects.keys())
        object_relationships = get_object_relationships(
            observation, env.objects, available_predicates, use_hand_state=False
        )
        object_relationships = [str(prop) for prop in object_relationships]
        all_prior_object_relationships.append(object_relationships)
        lm_cfg.engine = "code-davinci-002"

        goal_props_predicted, lm_cache = get_goal_from_lm(
            env.instruction,
            objects,
            object_relationships,
            pddl_domain,
            pddl_problem,
            examples=examples,
            lm_cfg=lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
            verbose=verbose,
        )
        goal_props_ground_truth: List[str] = [str(goal) for goal in env.goal_propositions]
        save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

        if use_ground_truth_goal_props:
            goal_props_to_use = goal_props_ground_truth
        else:
            goal_props_to_use = goal_props_predicted

        goal_props_callable: List[predicates.Predicate] = get_callable_goal_props(
            goal_props_to_use, possible_props
        )

        print(colored(f"goal props predicted: {goal_props_predicted}", "blue"))
        print(colored(f"minimal goal props: {goal_props_ground_truth}", "blue"))

        done: bool = False
        reached_goal_prop: bool = False

        actions: List[str]
        env.record_start()

        step = 0
        while not done:
            # lm_cfg.engine = "text-davinci-003"  # 002 is bad at following instructions
            actions, lm_cache = get_next_actions_from_lm(
                env.instruction,
                goal_props_to_use,
                objects,
                object_relationships,
                all_prior_object_relationships,
                all_executed_actions,
                pddl_domain,
                pddl_problem,
                custom_in_context_example_robot_prompt="Top robot action sequence: ",
                custom_robot_prompt="Top 5 valid next actions (python list of primitives): ",
                examples=examples,
                lm_cfg=lm_cfg,
                auth=auth,
                lm_cache=lm_cache,
            )
            save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

            print(
                colored(
                    f"Executed actions: {list(map(str, all_executed_actions))}",
                    "green",
                )
            )
            print(colored(f"Potential actions: {actions}", "yellow"))
            env_lst = [env] * len(actions)
            potential_actions: List[table_primitives.Primitive] = [
                env.get_primitive_info(action, env)
                for (action, env) in zip(actions, env_lst)
            ]
            potential_actions_str: List[str] = [
                str(action).lower() for action in potential_actions
            ]
            # lm_cfg.engine = "text-davinci-002"
            # 003 isn't really influenced by the in-context examples at all -
            # --- pick(hook) is really low
            lm_action_scores, lm_cache = get_action_scores_from_lm(
                env.instruction,
                potential_actions_str,
                goal_props_to_use,
                objects,
                object_relationships,
                all_prior_object_relationships,
                all_executed_actions,
                pddl_domain,
                pddl_problem,
                examples=examples,
                lm_cfg=lm_cfg,
                auth=auth,
                lm_cache=lm_cache,
                lm_cache_file=lm_cache_file,
            )

            save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)
            policy_actions: np.ndarray = get_policy_actions(
                observation,
                potential_actions,
                planner.policies,
                planner.dynamics,
                device=tensors.device(device),
            )
            value_action_scores: List[float] = get_values(
                observation,
                policy_actions,
                potential_actions,
                planner.policies,
                planner.dynamics,
                device=tensors.device(device),
            )

            table_headers = ["Action", "LM", "Value", "Overall"]
            overall_scores = [
                lm_action_score * value_action_score
                for (lm_action_score, value_action_score) in zip(
                    lm_action_scores, value_action_scores
                )
            ]
            formatted_table_str = format_saycan_scoring_table(
                table_headers,
                potential_actions_str,
                lm_action_scores,
                value_action_scores,
                overall_scores,
            )
            best_action_idx = overall_scores.index(max(overall_scores))
            env.set_primitive(potential_actions[best_action_idx])

            custom_recording_text: str = (
                f"human: {env.instruction}\n"
                + f"goal_props: {goal_props_to_use}\n"
                + f"{formatted_table_str}\n"
            )

            custom_recording_text += get_printable_object_relationships_str(
                object_relationships, max_row_length=70
            )

            observation, reward, terminated, _, info = env.step(
                policy_actions[best_action_idx], custom_recording_text
            )
            step += 1

            print(
                f"action executed: {potential_actions[best_action_idx]}; reward: {reward}"
            )

            if is_satisfy_goal_props(
                goal_props_callable,
                prop_testing_objs,
                observation,
                use_hand_state=False,
            ):
                print(colored("goal props satisfied", "green"))
                done = True
                reached_goal_prop = True

            objects = list(env.objects.keys())
            object_relationships = get_object_relationships(
                observation, env.objects, available_predicates, use_hand_state=False
            )
            print(f"object relationships: {object_relationships}")
            object_relationships = [str(prop) for prop in object_relationships]
            all_executed_actions.append(potential_actions[best_action_idx])
            all_prior_object_relationships.append(object_relationships)
            if step == max_depth:
                done = True

        env.record_stop()
        if env.is_goal_state():
            print(colored("ground truth goal props satisfied", "green"))
            num_successes_on_ground_truth_goal_props += 1


        gif_path = (
            path
            / str(idx_iter)
            / ("use-gt-goals" if use_ground_truth_goal_props else "use-pred-goals")
            / f"execution_{'reached_goal_prop' if reached_goal_prop else 'fail'}.gif"
        )
        env.record_save(gif_path, reset=False)
        gif_path = (
            path
            / str(idx_iter)
            / ("use-gt-goals" if use_ground_truth_goal_props else "use-pred-goals")
            / f"execution_{'reached_goal_prop' if reached_goal_prop else 'fail'}.mp4"
        )
        env.record_save(gif_path, reset=True)
        env._recording_text = ""

    # Save planning results.
    with open(path / f"results_seed_{seed}.json", "w") as f:
        save_dict = {
            "args": {
                "planner_config": planner_config,
                "env_config": env_config,
                "policy_checkpoints": policy_checkpoints,
                "dynamics_checkpoint": dynamics_checkpoint,
                "device": device,
                "num_eval": num_eval,
                "path": str(path),
                "pddl_domain": pddl_domain,
                "pddl_problem": pddl_problem,
                "max_depth": max_depth,
                "timeout": timeout,
                "seed": seed,
                "verbose": verbose,
                "use_ground_truth_goal_props": use_ground_truth_goal_props,
            },
            "num_successes_on_predicted_goal_props": num_successes_on_predicted_goal_props,
            "success_rate_on_predicted_goal_props": (
                num_successes_on_predicted_goal_props / num_eval
            )
            * 100,
            "num_successes_on_ground_truth_goal_props": num_successes_on_ground_truth_goal_props,
            "success_rate_on_ground_truth_goal_props": (
                num_successes_on_ground_truth_goal_props / num_eval
            )
            * 100,
        }
        json.dump(save_dict, f, indent=4)


def main(args: argparse.Namespace) -> None:
    auth = authenticate(args.api_type, args.key_name)
    delattr(args, "key_name")
    eval_saycan(**vars(args), auth=auth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-config",
        "--planner",
        "-c",
        help="Path to planner config, used to load value functions (dynamics aren't used for saycan)",
    )
    parser.add_argument("--env-config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument("--scod-checkpoints", "-s", nargs="+", help="SCOD checkpoints")
    parser.add_argument("--dynamics-checkpoint", "-d", help="Dynamics checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    parser.add_argument(
        "--use-ground-truth-goal-props",
        "-gt-goal",
        type=int,
        default=0,
        help="Whether to use ground truth goal props instead of predicted goal props for termination and prompting",
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument(
        "--closed-loop", default=1, type=int, help="Run closed-loop planning"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--pddl-domain", help="Pddl domain")
    parser.add_argument("--pddl-problem", help="Pddl problem")
    parser.add_argument(
        "--max-depth", type=int, default=10, help="Task planning search depth"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Task planning timeout"
    )
    # EvaluationConfig
    parser.add_argument(
        "--lm_cache_file", type=str, default="lm_cache.pkl", help="LM cache file"
    )
    # PDDLConfig
    parser.add_argument(
        "--pddl-root-dir",
        type=str,
        default="configs/pybullet/envs/official/domains",
        help="PDDL root dir",
    )
    parser.add_argument(
        "--pddl-domain-name",
        type=str,
        default="hook_reach",
        help="PDDL domain",
        choices=["constrained_packing", "hook_reach"],
    )
    # PromptConfig
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Number of examples to use in in context prompt",
    )
    # LMConfig
    parser.add_argument(
        "--engine",
        type=str,
        default="curie",
        help="LM engine (curie or davinci or baggage or ada)",
    )
    parser.add_argument("--temperature", type=float, default=0, help="LM temperature")
    parser.add_argument(
        "--api_type", type=APIType, default=APIType.OPENAI, help="API to use"
    )
    parser.add_argument(
        "--key-name",
        type=str,
        choices=["personal-code", "personal-all", "helm"],
        default="personal-code",
        help="API key name to use",
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="LM max tokens")
    parser.add_argument("--logprobs", type=int, default=1, help="LM logprobs")

    args = parser.parse_args()

    main(args)
