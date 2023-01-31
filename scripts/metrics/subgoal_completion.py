from typing import Union, List, Set

import yaml
import json
import argparse
import pathlib
import symbolic
import numpy as np


def compute_plan(
    pddl,
    state: Set[str],
    timeout: float = 10.0,
    max_depth: int = 10,
    verbose: bool = False,
) -> List[str]:
    pddl.initial_state = state
    planner = symbolic.Planner(pddl, pddl.initial_state)
    bfs = symbolic.BreadthFirstSearch(
        planner.root, 
        max_depth=max_depth, 
        timeout=timeout, 
        verbose=verbose
    )
    for plan in bfs:
        break
    return plan


def compute_subgoal_completion(
    path: Union[str, pathlib.Path],
    domain_file: Union[str, pathlib.Path],
    problem_file: Union[str, pathlib.Path],
    timeout: float = 10.0,
    max_depth: int = 10,
    verbose: bool = False,
):
    # Compute optimal solution to the TAMP problem.
    pddl = symbolic.Pddl(domain_file, problem_file)
    optimal_plan = compute_plan(
        pddl, 
        pddl.initial_state,
        timeout=timeout,
        max_depth=max_depth,
        verbose=verbose
    )

    # Load save_compressedz.
    actions = []

    state = pddl.initial_state
    for action_call in actions:
        if not pddl.is_valid_action(state, action_call):
            break
        state = pddl.next_state(state, action_call)
    plan_to_go = compute_plan(
        pddl, 
        state,
        timeout=timeout,
        max_depth=max_depth,
        verbose=verbose
    )

    # Subgoal completion = 1 - len(plan_to_go) / len(optimal_plan)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to saved output file.")
    parser.add_argument("--domain-file", type=str, required=True, help="Path to PDDL domain file.")
    parser.add_argument("--problem-file", type=str, required=True, help="Path to PDDL problem file.")
    parser.add_argument("--timeout", type=float, help="Planning timeout.")
    parser.add_argument("--max-depth", type=int, help="Maximum search depth.")
    parser.add_argument("--verbose", action="store_true", help="Verbose planning.")
    compute_subgoal_completion(**vars(parser.parse_args()))
