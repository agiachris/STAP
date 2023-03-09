from typing import Union, List, Set

import os
import json
import argparse
import pathlib
import symbolic


def compute_plan(
    pddl,
    state: Set[str],
    timeout: float = 10.0,
    max_depth: int = 20,
    verbose: bool = False,
) -> List[str]:
    pddl.initial_state = state
    planner = symbolic.Planner(pddl, pddl.initial_state)
    bfs = symbolic.BreadthFirstSearch(
        planner.root, max_depth=max_depth, timeout=timeout, verbose=verbose
    )
    plan = None
    for _plan in bfs:
        plan = _plan
        break
    return plan


def compute_subgoal_completion(
    path: Union[str, pathlib.Path],
    domain_file: Union[str, pathlib.Path],
    timeout: float = 10.0,
    max_depth: int = 20,
    verbose: bool = False,
    save: bool = False,
) -> float:
    if isinstance(path, str):
        path = pathlib.Path(path)
    assert path.exists(), "Path to existing results does not exist."
    output_path = path.parent / (path.stem + "_subgoal.json")

    # Load planning result.
    with open(path, "r") as f:
        planning_result = json.load(f)

    # Get problem filepath.
    problem_postfix = os.path.split(domain_file)[-1].split("_")[0]
    task_path = pathlib.Path(planning_result["args"]["env_config"])
    problem_file = str(task_path.parent / (task_path.stem + f"_{problem_postfix}.pddl"))

    # Load TAMP problem.
    num_success = 0
    num_failure = 0
    failure_subgoal_completion = 0
    for run_log in planning_result["run_logs"]:
        if "inner_monologue" in str(path) or "saycan" in str(path):
            if run_log["success"]:
                run_log["subgoal_completion_rate"] = 1.0
                num_success += 1
                continue
        else:
            if run_log["reached_ground_truth_goal"]:
                run_log["subgoal_completion_rate"] = 1.0
                num_success += 1
                continue

        plan_cost = 0
        pddl = symbolic.Pddl(domain_file, problem_file)

        state = pddl.initial_state
        for action_call in run_log["executed_actions"]:
            action_call = action_call.lower()
            if "stop" in action_call or not pddl.is_valid_action(state, action_call):
                break
            state = pddl.next_state(state, action_call)
            plan_cost += 1

        plan_to_go = compute_plan(
            pddl, state, timeout=timeout, max_depth=max_depth, verbose=verbose
        )
        cost_to_go = len([node.action for node in plan_to_go][1:])

        if cost_to_go == 0:
            pddl = symbolic.Pddl(domain_file, problem_file)
            optimal_plan = compute_plan(
                pddl,
                pddl.initial_state,
                timeout=timeout,
                max_depth=max_depth,
                verbose=verbose,
            )
            optimal_cost = len([node.action for node in optimal_plan][1:])
            cost_diff = 1 if plan_cost <= optimal_cost else plan_cost - optimal_cost
            subgoal_rate = (optimal_cost - cost_diff) / optimal_cost

        else:
            subgoal_rate = plan_cost / (plan_cost + cost_to_go)

        num_failure += 1
        failure_subgoal_completion += subgoal_rate
        run_log["subgoal_completion_rate"] = subgoal_rate

    subgoal_completion_rate = (num_success + failure_subgoal_completion) / (
        num_success + num_failure
    )
    planning_result["subgoal_completion_rate"] = subgoal_completion_rate

    if save:
        with open(output_path, "w") as f:
            json.dump(planning_result, f, indent=2)

    return subgoal_completion_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Path to saved output file."
    )
    parser.add_argument(
        "--domain-file", type=str, required=True, help="Path to PDDL domain file."
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Planning timeout.")
    parser.add_argument(
        "--max-depth", type=int, default=20, help="Maximum search depth."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose planning.")
    parser.add_argument(
        "--save", action="store_true", help="Save the result with updated stats."
    )
    compute_subgoal_completion(**vars(parser.parse_args()))
