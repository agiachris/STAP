"""
Usage:

PYTHONPATH=. python scripts/eval/eval_saycan.py  --planner-config configs/pybullet/planners/ablation/policy_cem.yaml --env-config configs/pybullet/envs/official/domains/constrained_packing/tamp0.yaml 
--policy-checkpoints models/official/pick/select_model.pt models/official/place/select_model.pt models/official/pull/select_model.pt models/official/push/select_model.pt
--dynamics-checkpoint models/official/select_model/dynamics/best_model.pt
--pddl-domain configs/pybullet/envs/official/domains/constrained_packing/tamp0_domain.pddl --pddl-problem configs/pybullet/envs/official/domains/constrained_packing/tamp0_problem.pddl 
--max-depth 4 --timeout 10 --closed-loop 1 --num-eval 100 --path plots/20221105/decoupled_state/tamp_experiment/constrained_packing/tamp0 --verbose 0 --engine davinci --gui 1

# --policy-checkpoints models/20221105/decoupled_state/pick/ckpt_model_10.pt models/20221105/decoupled_state/place/ckpt_model_10.pt models/20221105/decoupled_state/pull/ckpt_model_10.pt 
# models/20221105/decoupled_state/push/ckpt_model_10.pt --dynamics-checkpoint models/20221105/decoupled_state/ckpt_model_10/dynamics/final_model.pt --seed 0 
"""

import functools
import random
from temporal_policies import agents, envs, planners
from temporal_policies.dynamics.base import Dynamics
from temporal_policies.envs.pybullet.table.objects import Object

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Union
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


def get_values(
    observation: np.ndarray,
    actions: np.ndarray,
    primitives: List[table_primitives.Primitive],
    policies: Sequence[agents.Agent],
    dynamics: Dynamics,
    device: torch.device,
) -> List[float]:
    """Get the value of the observation for each primitive."""
    value_fns = [
        policies[primitive.idx_policy].critic for primitive in primitives
    ]
    policy_fns = [
        policies[primitive.idx_policy] for primitive in primitives
    ]
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
            policy_state = decode_fns[i](t_observation)
            # if i == 0:
            #     print(f't_observation: {t_observation}')
            #     print(f'primitives[i]: {primitives[i]}')
            #     print(f'policy_state: {policy_state}')
            print("actions[i]: ", actions[i])
            print("policy_fns[i].actor.predict(policy_state).cpu().numpy(): ", policy_fns[i].actor.predict(policy_state).cpu().numpy())
            # assert (actions[i] == policy_fns[i].actor.predict(policy_state).cpu().numpy()).all(), "action is not the same as the one predicted by the policy"
            q_s_a = value_fns[i].predict(policy_state, tensors.from_numpy(actions[i], device=device))
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
    policy_fns = [
        policies[primitive.idx_policy] for primitive in primitives
    ]

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
            policy_state = decode_fns[i](t_observation)
            action = policy_fns[i].actor.predict(policy_state)
            actions.append(action.cpu().numpy())
            # if i == 0:
            #     print(f't_observation: {t_observation}')
            #     print(f'primitives[i]: {primitives[i]}')
            #     print(f'policy_state: {policy_state}')
    return actions


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
    max_depth: int = 5,
    timeout: float = 10.0,
    verbose: bool = False,
    load_path: Optional[Union[str, pathlib.Path]] = None,
    seed: Optional[int] = None,
    gui: Optional[int] = None,
    lm_cache_file: Optional[Union[str, pathlib.Path]] = None,
    n_examples: Optional[int] = 1,
    custom_in_context_example_robot_prompt: str = "Top 1 robot action sequences: ",
    custom_in_context_example_robot_format: str = "python list of lists",
    custom_robot_prompt: str = "Top 2 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: str = "python list of lists",
    engine: Optional[str] = None,
    temperature: Optional[int] = 0,
    logprobs: Optional[int] = 1,
    api_type: Optional[APIType] = APIType.HELM,
    max_tokens: Optional[int] = 100,
    auth: Optional[Authentication] = None,
):
    INSTRUCTION = "Put the red box on the rack"

    examples = get_examples_from_json_dir(pddl_root_dir + "/" + pddl_domain_name)
    print(len(examples), n_examples)
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
    
    prop_testing_objs: Dict[str, Object] = get_prop_testing_objs(env)

    available_predicates: List[str] = ["on", "inhand"]
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

    actions: List[str]
    while not done:
        actions, lm_cache = get_next_actions_from_lm(
            INSTRUCTION,
            goal_props_predicted,
            objects,
            object_relationships,
            all_prior_object_relationships,
            all_executed_actions,
            pddl_domain,
            pddl_problem,
            custom_in_context_example_robot_prompt="Top robot action sequence: ",
            custom_in_context_example_robot_format="python list",
            custom_robot_prompt="6 valid next actions (python list): ",
            custom_robot_action_sequence_format="python list",
            examples=examples,
            lm_cfg=lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
        )
        save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

        env_lst = [env] * len(actions)
        potential_actions: List[table_primitives.Primitive] = [
            env.get_primitive_info(action, env)
            for (action, env) in zip(actions, env_lst)
        ]
        potential_actions_str: List[str] = [str(action).lower() for action in potential_actions]
        lm_action_scores, lm_cache = get_action_scores_from_lm(
            INSTRUCTION,
            potential_actions_str,
            goal_props_predicted,
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
        )
        save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

        policy_actions: np.ndarray = get_policy_actions(
            observation, potential_actions, planner.policies, planner.dynamics, device=tensors.device(device)
        )
        print("policy actions: ", policy_actions)
        value_action_scores: List[float] = get_values(
            observation, policy_actions, potential_actions, planner.policies, planner.dynamics, device=tensors.device(device)
        )
        print("value action scores: ", value_action_scores)
        overall_scores: List[float] = [
            lm_action_scores[i] + value_action_scores[i]
            for i in range(len(potential_actions))
        ]  # the 'done' skill seems to have a hard-coded value according to the figures in saycan ...
        best_action_idx = overall_scores.index(max(overall_scores))
        env.set_primitive(potential_actions[best_action_idx])
        print("best action: ", potential_actions[best_action_idx])
        observation, reward, terminated, _, info = env.step(
            policy_actions[best_action_idx]
        )  # return done = True if LM's score on done is the highest?

        if is_satisfy_goal_props(
            goal_props_callable, prop_testing_objs, observation, use_hand_state=False
        ):
            print("goal props satisfied")
            done = True
        objects = list(env.objects.keys())
        object_relationships = get_object_relationships(
            observation, env.objects, available_predicates, use_hand_state=False
        )


def main(args: argparse.Namespace) -> None:
    # TODO(klin) add function that checks if we've reached the goal state?
    if args.api_type.value == APIType.OPENAI.value:
        api_key = "***REMOVED***"
    elif args.api_type.value == APIType.HELM.value:
        # read api_key from credentials.json file
        with open("credentials.json", "r") as f:
            api_key = json.load(f)["openaiApiKey"]
    else:
        raise ValueError("Invalid API type")
    auth = register_api_key(args.api_type, api_key)
    eval_saycan(**vars(args), auth=auth)


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
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--pddl-domain", help="Pddl domain")
    parser.add_argument("--pddl-problem", help="Pddl problem")
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
        default="constrained_packing",
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
        "--api_type", type=APIType, default=APIType.HELM, help="API to use"
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="LM max tokens")
    parser.add_argument("--logprobs", type=int, default=1, help="LM logprobs")

    args = parser.parse_args()

    main(args)
