"""
Usage:

PYTHONPATH=. python scripts/eval/eval_saycan.py
--planner-config configs/pybullet/planners/policy_cem.yaml 
--env-config configs/pybullet/envs/t2m/official/tasks/task0.yaml
--policy-checkpoints
    models/20230121/policy/pick_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt
    models/20230121/policy/place_value_sched-cos_iter-5M_sac_ens_value/final_model/final_model.pt
    models/20230120/policy/pull_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt
    models/20230120/policy/push_value_sched-cos_iter-2M_sac_ens_value/final_model/final_model.pt
--dynamics-checkpoint
    models/20230121/dynamics/pick_place_pull_push_dynamics/best_model.pt
--pddl-domain configs/pybullet/envs/t2m/official/tasks/symbolic_domain.pddl
--pddl-problem configs/pybullet/envs/t2m/official/tasks/task0_symbolic.pddl
--verbose 0 --engine davinci --gui 0
--path plots/20230121-latest-models/eval_saycan
--key-name personal-all
--max-depth 7
--n-examples 10
--lm-verbose 1
--gui 1
"""

import functools
import random
from temporal_policies import agents, envs, planners
from temporal_policies.dynamics.base import Dynamics
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.networks import critics

import tqdm
import tabulate
from temporal_policies.networks.critics.base import Critic

from temporal_policies.planners.utils import get_printable_object_relationships_str

tabulate.PRESERVE_WHITESPACE = True
from tabulate import tabulate
from symbolic import parse_proposition

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
from temporal_policies.task_planners.goals import get_goal_from_lm

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
                if isinstance(value_fns[i], critics.EnsembleOODCritic):
                    ood_filter = 1 - value_fns[i].detect.float()
                    q_s_a = q_s_a * ood_filter
                # clip values between 0 and 1
                q_s_a = torch.clamp(q_s_a, 0, 1)
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
    custom_in_context_example_robot_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_stop"
    ] = "python_list_with_stop",
    custom_robot_prompt: str = "Top 2 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_stop"
    ] = "python_list_with_stop",
    engine: Optional[str] = None,
    temperature: Optional[int] = 0,
    logprobs: Optional[int] = 1,
    api_type: Optional[APIType] = APIType.HELM,
    max_tokens: Optional[int] = 100,
    auth: Optional[Authentication] = None,
    use_ground_truth_goal_props: bool = False,
    lm_verbose: bool = False,
):
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # these would perhaps belong in .../prompts/
    examples = get_examples_from_json_dir("configs/pybullet/envs/t2m/official/prompts/")

    for example in examples:
        example.custom_robot_action_sequence_format = (
            custom_robot_action_sequence_format
        )

    examples = random.sample(examples, n_examples)
    random.shuffle(examples)
    lm_cfg: LMConfig = LMConfig(
        engine=engine,
        temperature=temperature,
        logprobs=logprobs,
        echo=False,
        api_type=api_type,
        max_tokens=max_tokens,
    )
    # Load environment.
    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)

    # add task name to lm_cache_file
    if lm_cache_file is not None:
        lm_cache_file_suffix = pathlib.Path(lm_cache_file).suffix
        lm_cache_file = pathlib.Path(lm_cache_file).stem
        lm_cache_file = (
            lm_cache_file + "_" + env.name + "_inner_monologue" + lm_cache_file_suffix
        )

    lm_cache: Dict[str, str] = load_lm_cache(pathlib.Path(lm_cache_file))

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

    pbar = tqdm.tqdm(
        seed_generator(num_eval, load_path), f"Evaluate {path.name}", dynamic_ncols=True
    )
    run_logs: List[Dict[str, Any]] = []

    # step = -1 if stopped on non-goal prop, else step = num steps to goal
    steps_to_success_via_pred_goal_props: List[int] = []
    steps_to_success_via_stop_score: List[int] = []

    for idx_iter, (seed, loaded_plan) in enumerate(pbar):
        goal_props_predicted: List[str]
        objects: List[str]
        object_relationships: List[str]
        object_relationships_history: List[List[str]] = []
        executed_actions: List[str] = []

        observation, info = env.reset(seed=seed)
        seed = info["seed"]
        print(f"seed: {seed}")

        # get goal props
        objects = list(env.objects.keys())
        object_relationships = get_object_relationships(
            observation, prop_testing_objs, available_predicates, use_hand_state=False
        )
        object_relationships = [str(prop) for prop in object_relationships]
        object_relationships_history.append(object_relationships)
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
            verbose=lm_verbose,
        )
        save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)
        goal_props_ground_truth: List[str] = [
            str(goal) for goal in env.goal_propositions
        ]

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

        actions: List[str]
        env.record_start()

        step = 0
        while not done:
            # lm_cfg.engine = "text-davinci-003"  # 002 is bad at following instructions
            # for generating possible actions, don't include stop()
            lm_verbose = True
            actions, lm_cache = get_next_actions_from_lm(
                env.instruction,
                goal_props_to_use,
                objects,
                object_relationships,
                object_relationships_history,
                executed_actions,
                pddl_domain,
                pddl_problem,
                custom_in_context_example_robot_prompt="Top robot action sequence: ",
                custom_in_context_example_robot_format="python_list",
                custom_robot_prompt="Top 5 valid next actions (python list of primitives): ",
                custom_robot_action_sequence_format="python_list",
                examples=examples,
                lm_cfg=lm_cfg,
                auth=auth,
                lm_cache=lm_cache,
                verbose=lm_verbose,
            )

            print(
                colored(
                    f"Executed actions: {list(map(str, executed_actions))}",
                    "green",
                )
            )
            print(colored(f"Potential actions: {actions}", "yellow"))
            # remove actions
            env_lst = [env] * len(actions)
            potential_actions: List[table_primitives.Primitive] = []
            for action, env in zip(actions, env_lst):
                try:
                    potential_actions.append(env.get_primitive_info(action, env))
                except Exception as e:
                    print(f"Exception: {e}")
                    continue
            potential_actions_str: List[str] = [
                str(action).lower() for action in potential_actions
            ]
            if custom_robot_action_sequence_format == "python_list_with_stop":
                potential_actions_str.append("stop()")
            # 003 isn't really influenced by the in-context examples at all -
            # --- pick(hook) is really low
            # for scoring possible actions, include stop()!
            lm_action_scores, lm_cache = get_action_scores_from_lm(
                env.instruction,
                potential_actions_str,
                goal_props_to_use,
                objects,
                object_relationships,
                object_relationships_history,
                executed_actions,
                pddl_domain,
                pddl_problem,
                examples=examples,
                custom_in_context_example_robot_format=custom_robot_action_sequence_format,
                custom_robot_action_sequence_format=custom_robot_action_sequence_format,
                lm_cfg=lm_cfg,
                auth=auth,
                lm_cache=lm_cache,
                lm_cache_file=lm_cache_file,
                verbose=lm_verbose,
            )

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

            if custom_robot_action_sequence_format == "python_list_with_stop":
                stop_score = lm_action_scores[-1]
                policy_actions.append(np.zeros(1))
                if stop_score == max(lm_action_scores):
                    value_action_scores.append(10)
                    done = True
                    print(colored("Stopping because stop() score is highest", "green"))
                else:
                    value_action_scores.append(0.5)

            if is_satisfy_goal_props(
                goal_props_callable,
                prop_testing_objs,
                observation,
                use_hand_state=False,
            ):
                print(colored("used goal props satisfied", "green"))
                if env.is_goal_state():
                    print(colored("goal props satisfied", "green"))
                    steps_to_success_via_pred_goal_props.append(step)
                else:
                    print(colored("goal props not satisfied", "red"))
                    steps_to_success_via_pred_goal_props.append(-1)

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
            print(colored(formatted_table_str, "blue"))

            best_action_idx = overall_scores.index(max(overall_scores))
            if "stop()" == potential_actions_str[best_action_idx]:
                env.set_primitive(table_primitives.Stop(env))
                done = True
                env.wait_until_stable()
                # TODO(klin) debug why the observation isn't correct for the red box's location
                print(
                    colored(
                        "Stopping because stop() lm * value score is highest", "green"
                    )
                )
            else:
                env.set_primitive(potential_actions[best_action_idx])

            observation = env.get_observation()
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
                f"action executed: {potential_actions_str[best_action_idx]}; reward: {reward}"
            )

            objects = list(env.objects.keys())
            object_relationships = get_object_relationships(
                observation,
                prop_testing_objs,
                available_predicates,
                use_hand_state=False,
            )
            print(f"object relationships: {list(map(str, object_relationships))}")
            object_relationships = [str(prop) for prop in object_relationships]
            executed_actions.append(potential_actions_str[best_action_idx])
            object_relationships_history.append(object_relationships)
            if step == max_depth:
                done = True

        save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

        env.record_stop()
        success: bool = env.is_goal_state()
        if success:
            steps_to_success_via_stop_score.append(
                step - 1
            )  # -1 because take extra step at end for rendering
            print(colored("ground truth goal props satisfied", "green"))
        else:
            steps_to_success_via_stop_score.append(-1)

        gif_path = (
            path
            / str(idx_iter)
            / f"execution_{'reached_goal_prop' if success else 'fail'}.gif"
        )
        env.record_save(gif_path, reset=False)
        gif_path = (
            path
            / str(idx_iter)
            / f"execution_{'reached_goal_prop' if success else 'fail'}.mp4"
        )
        env.record_save(gif_path, reset=True)
        env._recording_text = ""

        run_log = {
            "seed": seed,
            "success": success,
            "num_steps": step,
            "executed_actions": executed_actions,
            "object_relationships_history": object_relationships_history,
        }
        run_logs.append(run_log)

        # Save planning results.
        path.mkdir(parents=True, exist_ok=True)
        success_rate = sum([run_log["success"] for run_log in run_logs]) / len(run_logs)

        # Save planning results.
        with open(path / f"results_idx_iter_{idx_iter}_seed_{seed}.json", "w") as f:
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
                    "termination_method": "saycan_scoring",
                },
                "task_name": env.name,
                "task_file": str(pathlib.Path(env_config).name),
                "success_rate": success_rate,
                "goal_props_predicted": goal_props_predicted,
                "instruction": env.instruction,
                "run_logs": run_logs,
            }
            json.dump(save_dict, f, indent=2)


def main(args: argparse.Namespace) -> None:
    if args.api_type == "helm":
        args.api_type = APIType.HELM
    elif args.api_type == "openai":
        args.api_type = APIType.OPENAI
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
        default=10,
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
    parser.add_argument("--api-type", type=str, default="helm", help="API to use")
    parser.add_argument(
        "--key-name",
        type=str,
        choices=[
            "personal-code",
            "personal-all",
            "personal-raoak",
            "personal-m",
            "helm",
        ],
        default="personal-code",
        help="API key name to use",
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="LM max tokens")
    parser.add_argument("--logprobs", type=int, default=1, help="LM logprobs")
    parser.add_argument(
        "--custom_in_context_example_robot_format",
        type=str,
        default="python_list_with_stop",
        help="Custom in context example robot format",
    )
    parser.add_argument(
        "--custom_robot_action_sequence_format",
        type=str,
        default="python_list_with_stop",
        help="Custom in context example human format",
    )
    parser.add_argument(
        "--lm-verbose", type=int, default=0, help="Print out LM verbose flag"
    )
    args = parser.parse_args()

    main(args)
