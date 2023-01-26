"""
Usage:

PYTHONPATH=. python scripts/eval/eval_ilm_tamp.py
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
--verbose 0 --engine davinci --gui 0 --visualize-planning 1
--path plots/20230117/ilm_tamp/
--num-eval 3
--n-examples 5
"""

import json
import random
from temporal_policies import envs, planners
from temporal_policies.envs.pybullet.table.objects import Object

import tqdm
from termcolor import colored
import argparse
import pathlib
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import torch
import numpy as np
from helm.common.authentication import Authentication
from configs.base_config import LMConfig
from symbolic import parse_proposition

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
)

from temporal_policies.task_planners.beam_search import (
    BeamSearchAlgorithm,
    BeamSearchProblem,
    Node,
)


def eval_ilm_tamp(
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
    max_depth: int = 5,
    timeout: float = 10.0,
    verbose: bool = False,
    load_path: Optional[Union[str, pathlib.Path]] = None,
    seed: Optional[int] = 0,
    gui: Optional[int] = None,
    lm_cache_file: Optional[Union[str, pathlib.Path]] = None,
    n_examples: Optional[int] = 5,
    custom_in_context_example_robot_prompt: str = "Top 1 robot action sequences: ",
    custom_in_context_example_robot_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_stop"
    ] = "python_list",
    custom_robot_prompt: str = "Top 2 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_stop"
    ] = "python_list",
    engine: Optional[str] = None,
    temperature: Optional[int] = 0,
    logprobs: Optional[int] = 1,
    api_type: Optional[APIType] = APIType.HELM,
    max_tokens: Optional[int] = 100,
    auth: Optional[Authentication] = None,
    visualize_planning: bool = False,
    use_ground_truth_goal_props: bool = False,
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
    # lm_cfg.engine = "text-davinci-003"
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

    pbar = tqdm.tqdm(
        seed_generator(num_eval, load_path), f"Evaluate {path.name}", dynamic_ncols=True
    )

    # steps to success when stopping search + execution
    # via the predicted goal props
    steps_to_success_via_pred_goal_props: List[int] = []
    # steps to success when stopping search + execution
    # via asking the LLM to predict the next action and stopping if it predicts stop()
    steps_to_success_via_stop_prediction: List[int] = []

    for idx_iter, (seed, loaded_plan) in enumerate(pbar):
        goal_props_predicted: List[str]
        objects: List[str]
        object_relationships: List[str]
        all_prior_object_relationships: List[List[str]] = []
        all_executed_actions: List[str] = []

        observation, info = env.reset(seed=seed)
        seed = info["seed"]
        print(f"seed: {seed}")

        # get goal props
        objects = list(env.objects.keys())
        object_relationships = get_object_relationships(
            observation, env.objects, available_predicates, use_hand_state=False
        )
        object_relationships = [str(prop) for prop in object_relationships]
        all_prior_object_relationships.append(object_relationships)
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
        )
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
        recording_id: Optional[int] = None

        beam_search_algorithm = BeamSearchAlgorithm(
            max_beam_size=1, max_depth=max_depth, num_successors_per_node=5
        )

        if is_satisfy_goal_props(
            goal_props_callable,
            prop_testing_objs,
            observation,
            use_hand_state=False,
        ):
            print(colored("goal props satisfied", "green"))
            done = True

        step = 0
        while not done:
            beam_search_problem = BeamSearchProblem(
                env.instruction,
                goal_props_predicted,
                observation,
                planner,
                env,
                available_predicates,
                prop_testing_objs,
                goal_props_callable,
                pddl_domain,
                pddl_problem,
                examples,
                lm_cfg,
                all_prior_object_relationships,
                all_executed_actions,
                auth,
                lm_cache,
                lm_cache_file,
            )
            successful_action_nodes: List[Node] = beam_search_algorithm.solve(
                beam_search_problem,
                visualize=visualize_planning,
                visualize_path=path / str(idx_iter),
            )
            idx_best = np.argmax(
                [
                    node.motion_plan_post_optimization.values.prod()
                    for node in successful_action_nodes
                ]
            )
            successful_action_node = successful_action_nodes[idx_best]
            best_motion_plan = successful_action_node.motion_plan_post_optimization
            best_task_plan = successful_action_node.action_skeleton_as_primitives

            # Execute one step.
            action = best_motion_plan.actions[0]
            assert isinstance(action, np.ndarray)
            state = best_motion_plan.states[0]
            assert isinstance(state, np.ndarray)
            value = best_motion_plan.values[0]
            env.set_primitive(best_task_plan[0])

            if recording_id is None:
                recording_id = env.record_start()
            else:
                env.record_start(recording_id)
            observation, reward, _, _, _ = env.step(
                action, successful_action_node.custom_recording_text_sequence[0]
            )
            env.record_stop(recording_id)

            step += 1

            print(f"action executed: {str(best_task_plan[0])}; reward: {reward}")
            if is_satisfy_goal_props(
                goal_props_callable,
                prop_testing_objs,
                observation,
                use_hand_state=False,
            ):
                print(colored("predicted goal props satisfied", "green"))
                success = True
                done = True

            objects = list(env.objects.keys())
            object_relationships = get_object_relationships(
                observation, env.objects, available_predicates, use_hand_state=False
            )
            object_relationships = [str(prop) for prop in object_relationships]
            all_executed_actions.append(str(best_task_plan[0]))
            all_prior_object_relationships.append(object_relationships)
            if step == max_depth:
                done = True

        success: bool = env.is_goal_state()
        if success:
            steps_to_success_via_pred_goal_props.append(
                step - 1
            )  # -1 because take extra step at end for rendering
            print(colored("Success! Ground truth goal props satisfied.", "green"))
        else:
            steps_to_success_via_pred_goal_props.append(-1)

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

    save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)
    # Save planning results.
    path.mkdir(parents=True, exist_ok=True)

    num_successes_via_pred_goal_props = int(
        (np.array(steps_to_success_via_pred_goal_props) != -1).sum()
    )
    num_successes_via_pred_stop = int(
        (np.array(steps_to_success_via_stop_prediction) <= max_depth).sum()
    )

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
            "num_successes_via_pred_stop": num_successes_via_pred_stop,
            "success_rate_via_pred_stop": (num_successes_via_pred_stop / num_eval)
            * 100,
            "num_successes_via_pred_goal_props": num_successes_via_pred_goal_props,
            "success_rate_via_pred_goal_props": (
                num_successes_via_pred_goal_props / num_eval
            )
            * 100,
        }
        json.dump(save_dict, f, indent=4)

    # Print results.
    print(
        colored(
            f"Success rate: {num_successes_via_pred_goal_props / num_eval * 100}",
            "green",
        )
    )


def main(args: argparse.Namespace) -> None:
    if args.key_name == "helm":
        args.api_type = APIType.HELM
    auth = authenticate(args.api_type, args.key_name)
    assert (
        "code" not in args.key_name
    ), "Please use a non-code only key since get_successors uses text-davinci-003."
    delattr(args, "key_name")
    eval_ilm_tamp(**vars(args), auth=auth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-config", "--planner", "-c", help="Path to planner config"
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
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument(
        "--closed-loop", default=1, type=int, help="Run closed-loop planning"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument(
        "--max-depth", type=int, default=5, help="Task planning search depth"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Task planning timeout"
    )
    # EvaluationConfig
    parser.add_argument(
        "--lm_cache_file", type=str, default="lm_cache.pkl", help="LM cache file"
    )
    parser.add_argument("--pddl-domain", help="Pddl domain file")
    parser.add_argument("--pddl-problem", help="Pddl problem file")
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
        "--api-type", type=APIType, default=APIType.OPENAI, help="API to use"
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
    parser.add_argument(
        "--visualize-planning", type=int, default=0, help="Visualize planning process"
    )
    args = parser.parse_args()

    main(args)
