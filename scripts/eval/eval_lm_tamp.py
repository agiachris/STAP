"""
Usage:

PYTHONPATH=. python scripts/eval/eval_lm_tamp.py  
--planner-config configs/pybullet/planners/policy_cem.yaml
--env-config configs/pybullet/envs/t2m/official/tasks/task0.yaml
--policy-checkpoints
    models/20230118/policy/pick/final_model/best_model.pt
    models/20230118/policy/place/final_model/best_model.pt
    models/20230118/policy/pull/final_model/best_model.pt
    models/20230118/policy/push/final_model/best_model.pt
--dynamics-checkpoint
    models/20230118/dynamics/pick_place_pull_push_dynamics/best_model.pt
--pddl-domain configs/pybullet/envs/t2m/official/tasks/symbolic_domain.pddl
--pddl-problem configs/pybullet/envs/t2m/official/tasks/task0_symbolic.pddl
--seed 0
--pddl-domain-name hook_reach
--max-depth 8 --timeout 10 --closed-loop 1 --num-eval 10
--path plots/20230120-new-models/eval_lm_tamp/ --verbose 0 --engine davinci
--n-examples 5
--key-name personal-all
"""

import argparse
import copy
import json
import pathlib
import random
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence, Union

from termcolor import colored
import numpy as np
from scripts.eval.eval_saycan import (
    format_saycan_scoring_table,
    get_policy_actions,
    get_values,
)
from temporal_policies.planners.utils import get_printable_object_relationships_str
import torch
import symbolic
import tqdm
from helm.common.authentication import Authentication
from configs.base_config import LMConfig
from symbolic import parse_proposition

from temporal_policies import dynamics, envs, planners
from temporal_policies.envs.pybullet.table import (
    predicates,
    primitives as table_primitives,
)
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.evaluation.utils import (
    get_callable_goal_props,
    get_object_relationships,
    get_possible_props,
    get_prop_testing_objs,
    get_task_plan_primitives_instantiated,
    is_satisfy_goal_props,
    seed_generator,
)
from temporal_policies.task_planners.goals import get_goal_from_lm
from temporal_policies.task_planners.lm_data_structures import (
    APIType,
)
from temporal_policies.task_planners.lm_agent import LMPlannerAgent
from temporal_policies.task_planners.task_plans import (
    get_action_scores_from_lm,
    get_next_actions_from_lm,
)
from temporal_policies.utils import recording, tensors, timing

from temporal_policies.task_planners.lm_utils import (
    authenticate,
    get_examples_from_json_dir,
    load_lm_cache,
    save_lm_cache,
)


def eval_lm_tamp(
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
    custom_in_context_example_robot_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_stop"
    ] = "python_list",
    custom_robot_prompt: str = "Top 5 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: str = "python_list_of_lists",
    engine: Optional[str] = None,
    temperature: Optional[int] = 0,
    logprobs: Optional[int] = 1,
    api_type: Optional[APIType] = APIType.HELM,
    max_tokens: Optional[int] = 200,
    auth: Optional[Authentication] = None,
    termination_method: Literal[
        "goal_prop", "pred_instr_achieved"
    ] = "pred_instr_achieved",
    use_ground_truth_goal_props: bool = False,
) -> None:
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    examples = get_examples_from_json_dir("configs/pybullet/envs/t2m/official/prompts/")
    examples = random.sample(examples, n_examples)
    random.shuffle(examples)

    lm_cfg = LMConfig(
        engine=engine,
        temperature=temperature,
        logprobs=logprobs,
        echo=False,
        api_type=api_type,
        max_tokens=max_tokens,
    )
    lm_cache: Dict[str, str] = load_lm_cache(pathlib.Path(lm_cache_file))
    timer = timing.Timer()

    # Load environment.
    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)

    prop_testing_objs: Dict[str, Object] = get_prop_testing_objs(env)

    # Load planner.
    planner = planners.load(
        config=planner_config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        scod_checkpoints=scod_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    # first append the termination_method to the end of the path with making a subdirectory
    path = path + "_" + termination_method
    path = pathlib.Path(path) / env.name

    prop_testing_objs: Dict[str, Object] = get_prop_testing_objs(env)
    available_predicates: List[str] = [
        parse_proposition(prop)[0] for prop in env.supported_predicates
    ]
    possible_props: List[predicates.Predicate] = get_possible_props(
        env.objects, available_predicates
    )

    num_successes_on_ground_truth_goal_props: int = 0

    goal_props_predicted: List[str] = None
    goal_props_ground_truth: List[str] = [str(goal) for goal in env.goal_propositions]

    pbar = tqdm.tqdm(
        seed_generator(num_eval, load_path), f"Evaluate {path.name}", dynamic_ncols=True
    )
    run_logs: List[Dict[str, Any]] = []
    for idx_iter, (seed, loaded_plan) in enumerate(pbar):
        goal_props_predicted: List[str]
        objects: List[str]
        object_relationships: List[str]
        object_relationships_history: List[List[str]] = []
        executed_actions: List[str] = []
        done: bool = False

        observation, info = env.reset(seed=seed)
        seed = info["seed"]
        print(f"seed: {seed}")

        env_state = env.get_state()

        task_plans = []
        motion_plans = []
        motion_planner_times = []
        available_predicates: List[str] = [
            parse_proposition(prop)[0] for prop in env.supported_predicates
        ]
        objects: List[str] = list(env.objects.keys())
        object_relationships = get_object_relationships(
            observation, env.objects, available_predicates, use_hand_state=False
        )
        object_relationships_history.append(object_relationships)
        possible_props: List[predicates.Predicate] = get_possible_props(
            env.objects, available_predicates
        )

        if goal_props_predicted is None:
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
                verbose=True,
            )
            goal_props_callable: List[predicates.Predicate] = get_callable_goal_props(
                goal_props_predicted, possible_props
            )

        # TODO(klin): check if satisfies goal at first timestep

        print(colored(f"goal props predicted: {goal_props_predicted}", "blue"))
        print(colored(f"minimal goal props: {goal_props_ground_truth}", "blue"))

        recording_id: int = 100

        lm_cfg.max_tokens = 500
        lm_agent = LMPlannerAgent(
            instruction=env.instruction,
            scene_objects=objects,
            goal_props_predicted=goal_props_predicted,
            examples=examples,
            lm_cfg=lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
            pddl_domain_file=pddl_domain,
            pddl_problem_file=pddl_problem,
        )

        step: int = 0
        while not done:
            fallback_to_scoring: bool = True
            generated_task_plans = lm_agent.get_task_plans(
                object_relationships_history=object_relationships_history,
                executed_actions=executed_actions,
                custom_in_context_example_robot_prompt=custom_in_context_example_robot_prompt,
                custom_in_context_example_robot_format=custom_in_context_example_robot_format,
                custom_robot_prompt=custom_robot_prompt,
                custom_robot_action_sequence_format=custom_robot_action_sequence_format,
            )
            print(colored(f"generated task plans: {generated_task_plans}", "blue"))
            action_skeletons_instantiated = get_task_plan_primitives_instantiated(
                generated_task_plans, env
            )

            print(
                f"len(action_skeletons_instantiated): {len(action_skeletons_instantiated)}"
            )
            for i, action_skeleton in enumerate(action_skeletons_instantiated):
                print(f"action_skeleton: {[str(action) for action in action_skeleton]}")
                if len(action_skeleton) > max_depth:
                    print(f"action_skeleton too long. Skipping ...")
                    continue

                timer.tic("motion_planner")
                env.set_primitive(action_skeleton[0])
                plan = planner.plan(env.get_observation(), action_skeleton)
                t_motion_planner = timer.toc("motion_planner")

                task_plans.append(action_skeleton)
                motion_plans.append(plan)
                motion_planner_times.append(t_motion_planner)

                # Reset env for oracle value/dynamics.
                if isinstance(planner.dynamics, dynamics.OracleDynamics):
                    env.set_state(env_state)

                if "greedy" in str(planner_config):
                    break

            # Filter out plans that do not reach the goal.
            goal_reaching_task_plans = []
            goal_reaching_motion_plans = []
            goal_reaching_task_p_successes = []
            for task_plan, motion_plan in zip(task_plans, motion_plans):
                # find first timestep where the goal is reached for the current TAMP
                plan_object_relationships_history = [
                    list(x) for x in object_relationships_history
                ]
                plan_executed_actions = copy.deepcopy(executed_actions)
                for i in range(len(motion_plan.actions)):
                    new_state = motion_plan.states[i + 1]
                    assert not np.isnan(new_state).any()
                    new_object_relationships = get_object_relationships(
                        new_state,
                        env.objects,
                        available_predicates,
                        use_hand_state=False,
                    )
                    if termination_method == "pred_instr_achieved":
                        plan_object_relationships_history.append(
                            new_object_relationships
                        )
                        plan_executed_actions.append(str(task_plan[i]).lower())
                        next_action_str = lm_agent.get_next_action_str(
                            plan_object_relationships_history,
                            plan_executed_actions,
                            in_context_example_robot_format="python_list",
                            robot_prompt="Instruction achieved (True/False): ",
                            verbose=False,
                        )
                        lm_cache = lm_agent.lm_cache
                        is_end_lm: bool = "stop" in next_action_str[: len("stop()") + 5]
                    elif termination_method == "goal_prop":
                        is_end_lm: bool = is_satisfy_goal_props(
                            goal_props_callable,
                            prop_testing_objs,
                            env.get_observation(),  # not the last state, which is nan
                            use_hand_state=False,
                        )
                    else:
                        raise NotImplementedError("Unknown termination method")

                    is_geom_feasible: bool = all(
                        value > 0.45 for value in motion_plan.values[:i]
                    )
                    if is_end_lm and is_geom_feasible:
                        goal_reaching_task_plans.append(task_plan[: i + 1])
                        shortened_motion_plan = copy.deepcopy(motion_plan)
                        shortened_motion_plan.actions = motion_plan.actions[: i + 1]
                        goal_reaching_motion_plans.append(shortened_motion_plan)
                        p_success = np.prod(
                            [value for value in motion_plan.values[: i + 1]]
                        )
                        goal_reaching_task_p_successes.append(p_success)
                        print(
                            colored(
                                f"Plan reaches goal at action {i} with probability {p_success}",
                                "green",
                            )
                        )
                        print(
                            f"task_plan: {[str(action) for action in task_plan[:i + 1]]}"
                        )
                        print(
                            colored(
                                "Found a plan with each primitive p_success > 0.5",
                                "green",
                            )
                        )
                        fallback_to_scoring = False
                        break
                    elif not is_geom_feasible:
                        print(
                            f"Could not find a plan with each primitive p_success > 0.5"
                        )
                        break

            save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

            idx_best = (
                np.argmax(goal_reaching_task_p_successes)
                if len(goal_reaching_task_p_successes) > 0
                else 0
            )
            if len(goal_reaching_task_p_successes) > 0:
                best_task_plan = goal_reaching_task_plans[idx_best]
                best_motion_plan = goal_reaching_motion_plans[idx_best]

                # Execute one step.
                action = best_motion_plan.actions[0]
                assert isinstance(action, np.ndarray)
                state = best_motion_plan.states[0]
                assert isinstance(state, np.ndarray)
                p_success = best_motion_plan.p_success
                value = best_motion_plan.values[0]
                env.set_primitive(best_task_plan[0])

                env.record_start(recording_id)
                _, reward, _, _, _ = env.step(action)

                # Run closed-loop planning on the rest.
                rewards, plan, t_planner = planners.run_closed_loop_planning(
                    env, best_task_plan[1:], planner, timer
                )
                env.record_stop(recording_id)
                step += len(best_task_plan)

                rewards = np.append(reward, rewards)
                plan = planners.PlanningResult(
                    actions=np.concatenate((action[None, ...], plan.actions), axis=0),
                    states=np.concatenate((state[None, ...], plan.states), axis=0),
                    p_success=p_success * plan.p_success,
                    values=np.append(value, plan.values),
                )
                # Save recording.
                gif_path = path / f"planning_{idx_iter}.gif"
                if (rewards == 0.0).any():
                    gif_path = (
                        gif_path.parent / f"{gif_path.name}_fail{gif_path.suffix}"
                    )
                env.record_save(gif_path, reset=False)

                executed_actions.extend(
                    [str(action).lower() for action in best_task_plan]
                )
                object_relationships_history.extend(
                    [
                        get_object_relationships(
                            state,
                            prop_testing_objs,
                            available_predicates,
                            use_hand_state=False,
                        )
                        for state in plan.states[1:]
                    ]
                )  # skip the first plan.states since that observation is already in the history
                print(f"executed_actions: {len(executed_actions)}")
                print(
                    f"object_relationships_history: {len(object_relationships_history)}"
                )

                if t_planner is not None:
                    motion_planner_times += t_planner

                if termination_method == "pred_instr_achieved":
                    done = "stop" in next_action_str[: len("stop()") + 5]
                    if done:
                        fallback_to_scoring = False
                        print(
                            colored("Stopping because stop() is next action", "green")
                        )
                elif termination_method == "goal_prop":
                    done = is_satisfy_goal_props(
                        goal_props_callable,
                        prop_testing_objs,
                        env.get_observation(),  # not the last state, which is nan
                        use_hand_state=False,
                    )
                    if done:
                        fallback_to_scoring = False
                        print(
                            colored(
                                "Stopping because states satisfies predicted goal props",
                                "green",
                            )
                        )
                else:
                    raise NotImplementedError(
                        f"Unknown termination method: {termination_method}"
                    )

                # after doing closed loop planning, we still need to check if the goal is reached
                # however, for the sake of evaluation, we will not do this check since
                # it's likely that re-planning will exceed max depth anyway
                done = True
                fallback_to_scoring = False

            if fallback_to_scoring:
                print(colored("No plan reaches the goal", "red"))
                # implement SayCan-like step here
                actions, lm_cache = get_next_actions_from_lm(
                    env.instruction,
                    goal_props_predicted,
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
                    verbose=False,
                )
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
                lm_action_scores, lm_cache = get_action_scores_from_lm(
                    env.instruction,
                    potential_actions_str,
                    goal_props_predicted,
                    objects,
                    object_relationships,
                    object_relationships_history,
                    executed_actions,
                    pddl_domain,
                    pddl_problem,
                    examples=examples,
                    custom_in_context_example_robot_format="python_list_with_stop",
                    custom_robot_action_sequence_format="python_list_with_stop",
                    lm_cfg=lm_cfg,
                    auth=auth,
                    lm_cache=lm_cache,
                    lm_cache_file=lm_cache_file,
                    verbose=False,
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
                        print(
                            colored("Stopping because stop() score is highest", "green")
                        )
                    else:
                        value_action_scores.append(0.5)

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
                    print(
                        colored(
                            "Stopping because stop() lm * value score is highest",
                            "green",
                        )
                    )
                else:
                    env.set_primitive(potential_actions[best_action_idx])

                observation = env.get_observation()
                custom_recording_text: str = (
                    f"human: {env.instruction}\n"
                    + f"goal_props: {goal_props_predicted}\n"
                    + f"{formatted_table_str}\n"
                )

                custom_recording_text += get_printable_object_relationships_str(
                    object_relationships, max_row_length=70
                )
                env.record_start(recording_id)
                observation, reward, terminated, _, info = env.step(
                    policy_actions[best_action_idx], custom_recording_text
                )
                env.record_stop(recording_id)

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

                next_action_str = lm_agent.get_next_action_str(
                    object_relationships_history,
                    executed_actions,
                )
                lm_cache = lm_agent.lm_cache
                save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

                if termination_method == "pred_instr_achieved":
                    done = "stop" in next_action_str[: len("stop()") + 5]
                    if done:
                        print(
                            colored("Stopping because stop() is next action", "green")
                        )
                elif termination_method == "goal_prop":
                    done = is_satisfy_goal_props(
                        goal_props_callable,
                        prop_testing_objs,
                        env.get_observation(),  # not the last state, which is nan
                        use_hand_state=False,
                    )
                    if done:
                        print(
                            colored(
                                "Stopping because states satisfies predicted goal props",
                                "green",
                            )
                        )
                else:
                    raise NotImplementedError(
                        f"Unknown termination method: {termination_method}"
                    )

                if step == max_depth:
                    done = True
                    print(colored("Stopping because max depth reached", "orange"))

            if env.is_goal_state():
                print(colored("ground truth goal props satisfied", "green"))
                num_successes_on_ground_truth_goal_props += 1

            # Print planning results.
            if verbose:
                print("success:", rewards.prod(), rewards)
                print(
                    "predicted success:",
                    motion_plans[idx_best].p_success,
                    motion_plans[idx_best].values,
                )
                if closed_loop:
                    print(
                        "visited predicted success:",
                        motion_plans[idx_best].p_visited_success,
                        motion_plans[idx_best].visited_values,
                    )
                for primitive, action in zip(
                    task_plans[idx_best], motion_plans[idx_best].actions
                ):
                    if isinstance(primitive, table_primitives.Primitive):
                        primitive_action = str(primitive.Action(action))
                        primitive_action = primitive_action.replace("\n", "\n  ")
                        print(
                            "-",
                            primitive,
                            primitive_action[primitive_action.find("{") :],
                        )
                    else:
                        print("-", primitive, action)
                print("discarded plans:")
                for idx_plan in range(len(task_plans)):
                    if idx_plan == idx_best:
                        continue
                    print(
                        "  - predicted success:",
                        motion_plans[idx_plan].p_success,
                        motion_plans[idx_plan].values,
                    )
                    for primitive, action in zip(
                        task_plans[idx_plan], motion_plans[idx_plan].actions
                    ):
                        if isinstance(primitive, table_primitives.Primitive):
                            primitive_action = str(primitive.Action(action))
                            primitive_action = primitive_action.replace(
                                "\n", "\n      "
                            )
                            print(
                                "    -",
                                primitive,
                                primitive_action[primitive_action.find("{") :],
                            )
                        else:
                            print("    -", primitive, action)
                continue

        env.record_stop()
        success = env.is_goal_state()
        if success:
            print(colored("ground truth goal props satisfied", "green"))
            num_successes_on_ground_truth_goal_props += 1

        gif_path = (
            path / str(idx_iter) / f"execution_{'success' if success else 'fail'}.gif"
        )
        env.record_save(gif_path, reset=False)
        gif_path = (
            path / str(idx_iter) / f"execution_{'success' if success else 'fail'}.mp4"
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

    # compute success rate but summing up the number of successes inside run_logs
    # and dividing by the number of runs
    success_rate = sum([run_log["success"] for run_log in run_logs]) / len(run_logs)

    path.mkdir(parents=True, exist_ok=True)
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
                "termination_method": termination_method,
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
    auth = authenticate(args.api_type)
    assert (
        "code" not in args.key_name
    ), "Please use a non-code only key since get_successors uses text-davinci-003."
    delattr(args, "key_name")
    eval_lm_tamp(**vars(args), auth=auth)


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
        "--max-depth", type=int, default=6, help="Task planning search depth"
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
    parser.add_argument(
        "--api_type", type=APIType, default=APIType.HELM, help="API to use"
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
        "--termination_method",
        type=str,
        default="pred_instr_achieved",
        help="LM termination method",
        choices=["pred_instr_achieved", "goal_prop"],
    )
    args = parser.parse_args()

    main(args)
