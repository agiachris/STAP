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
--verbose 0 --engine davinci --gui 1
"""

import functools
import random
from scripts.eval.eval_policies import query_policy_actor, query_policy_critic
from temporal_policies import agents, envs, planners
from temporal_policies.dynamics.base import Dynamics
from temporal_policies.envs.pybullet.table.objects import Object

import tabulate

tabulate.PRESERVE_WHITESPACE = True
from tabulate import tabulate

from termcolor import colored
import argparse
import pathlib
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import json
import symbolic


import torch
import numpy as np
from helm.common.authentication import Authentication
from configs.base_config import LMConfig

from temporal_policies.envs.pybullet.table import predicates
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.evaluation.utils import (
    get_callable_goal_props,
    get_goal_props_instantiated,
    get_object_relationships,
    get_possible_props,
    get_prop_testing_objs,
    is_satisfy_goal_props,
)
from temporal_policies.task_planners.goals import get_goal_from_lm, is_valid_goal_props

from temporal_policies.task_planners.lm_data_structures import APIType
from temporal_policies.task_planners.lm_utils import (
    authenticate,
    get_examples_from_json_dir,
    load_lm_cache,
    register_api_key,
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
from temporal_policies.task_planners.beam_search import BeamSearchAlgorithm, BeamSearchProblem


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
    # add escape code to format the color of the scores: if the score is closer to 1, set the color closer to green, if the score is closer to 0, set the color closer to red
    # rounded_lm_action_scores = [f'\033[38;2;{int(255 * (1 - lm_action_score))};{int(255 * lm_action_score)};0m{lm_action_score}\033[0m' for lm_action_score in rounded_lm_action_scores]
    # rounded_value_action_scores = [f'\033[38;2;{int(255 * (1 - value_action_score))};{int(255 * value_action_score)};0m{value_action_score}\033[0m' for value_action_score in rounded_value_action_scores]
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
):
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    INSTRUCTION = "Put the red box on the rack"
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

    prop_testing_objs: Dict[str, Object] = get_prop_testing_objs(env)

    available_predicates: List[str] = ["on", "inhand", "under"]
    possible_props: List[predicates.Predicate] = get_possible_props(
        env.objects, available_predicates
    )

    goal_props_predicted: List[str]
    objects: List[str]
    object_relationships: List[str]
    all_prior_object_relationships: List[List[str]] = []
    all_executed_actions: List[str] = []

    observation, info = env.reset()
    done = False

    # get goal props
    objects = list(env.objects.keys())
    object_relationships = get_object_relationships(
        observation, env.objects, available_predicates, use_hand_state=False
    )
    object_relationships = [str(prop) for prop in object_relationships]
    all_prior_object_relationships.append(object_relationships)
    goal_props_predicted, lm_cache = get_goal_from_lm(
        INSTRUCTION,
        objects,
        object_relationships,
        pddl_domain,
        pddl_problem,
        examples=examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
    )
    save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)
    goal_props_callable: List[predicates.Predicate] = get_callable_goal_props(
        goal_props_predicted, possible_props
    )
    print(f"goal props predicted: {goal_props_predicted}")

    beam_search_algorithm = BeamSearchAlgorithm(max_beam_size=1, max_depth=max_depth, num_successors_per_node=5)

    actions: List[str]
    env.record_start()
    # TODO(klin) load this from the yaml when ready
    max_steps = 10  # potentially task dependent and loadable from the yaml
    step = 0
    while not done:
        beam_search_problem = BeamSearchProblem(
            INSTRUCTION,
            goal_props_predicted,
            env.get_observation(),
            planner,
            env,
            available_predicates,
            prop_testing_objs,
            goal_props_callable,
            pddl_domain,
            pddl_problem,
            examples,
            lm_cfg,
            auth,
            lm_cache,
            lm_cache_file,
        )

        successful_action_nodes = beam_search_algorithm.solve(beam_search_problem, visualize=True)
        successful_action_node = successful_action_nodes[0]
        best_motion_plan = successful_action_node.motion_plan_post_optimization
        best_task_plan = successful_action_node.action_skeleton

        # Execute one step.
        action = best_motion_plan.actions[0]
        assert isinstance(action, np.ndarray)
        state = best_motion_plan.states[0]
        assert isinstance(state, np.ndarray)
        p_success = best_motion_plan.p_success
        value = best_motion_plan.values[0]
        env.set_primitive(best_task_plan[0])
        _, reward, _, _, _ = env.step(action, custom_recording_text)

        step += 1

        print(
            f"action executed: {potential_actions[best_action_idx]}; reward: {reward}"
        )

        if is_satisfy_goal_props(
            goal_props_callable, prop_testing_objs, observation, use_hand_state=False
        ):
            print("goal props satisfied")
            done = True

        objects = list(env.objects.keys())
        object_relationships = get_object_relationships(
            observation, env.objects, available_predicates, use_hand_state=False
        )
        object_relationships = [str(prop) for prop in object_relationships]
        all_executed_actions.append(potential_actions[best_action_idx])
        all_prior_object_relationships.append(object_relationships)
        if step == max_steps:
            done = True

    env.record_stop()
    gif_path = pathlib.Path(path) / f"saycan_execution_{step}_steps.gif"
    env.record_save(gif_path, reset=False)
    gif_path = pathlib.Path(path) / f"saycan_execution_{step}_steps.mp4"
    env.record_save(gif_path, reset=True)


def main(args: argparse.Namespace) -> None:
    auth = authenticate(args.api_type)
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
        "--max-depth", type=int, default=4, help="Task planning search depth"
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
        default=1,
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
    parser.add_argument("--max_tokens", type=int, default=100, help="LM max tokens")
    parser.add_argument("--logprobs", type=int, default=1, help="LM logprobs")

    args = parser.parse_args()

    main(args)
