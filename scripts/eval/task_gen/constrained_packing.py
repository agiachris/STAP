from typing import Any, List, Dict

import random
import argparse

from utils import (
    OBJECTS,
    LOCATIONS,
    LIFTED_OBJECTS,
    get_random_indices,
    get_lifted_task,
    substitute_vars,
    sort_propositions,
)


def constrained_packing_task(
    num_lifted_tasks: int, num_lifted_objects: int, poslimit: bool = False
) -> Dict[str, Any]:
    """Generate constrained packing lifted task from arguments."""
    lifted_task = {
        "num_tasks": num_lifted_tasks,
        "plan_skeleton": [
            x
            for p in zip(
                [f"pick({m}, table)" for m in LIFTED_OBJECTS[:num_lifted_objects]],
                [f"place({m}, rack)" for m in LIFTED_OBJECTS[:num_lifted_objects]],
            )
            for x in p
        ],
        "predicates": [f"free({m})" for m in LIFTED_OBJECTS[:num_lifted_objects]]
        + ["inworkspace(rack)"]
        + [f"inworkspace({m})" for m in LIFTED_OBJECTS[:num_lifted_objects]]
        + ["on(rack, table)"]
        + [f"on({m}, table)" for m in LIFTED_OBJECTS[:num_lifted_objects]],
    }
    if poslimit:
        lifted_task["predicates"].append("poslimit(rack)")
    return lifted_task


def main(objects: Dict[int, str], lifted_tasks: List[Dict[str, Any]]) -> None:
    """Randomly generate an even distribution of lifted evaluation tasks."""
    task_idx = 0
    num_tasks = sum(task["num_tasks"] for task in lifted_tasks)
    while task_idx < num_tasks:
        print(f"\nSampling lifted task: {task_idx}")

        # Sample ground objects for constrained packing problem
        lifted_task = get_lifted_task(task_idx, lifted_tasks)
        num_lifted_args = len(lifted_task["plan_skeleton"]) // 2
        arg_indices = random.sample(get_random_indices(objects), num_lifted_args)
        ground_args = [objects[idx] for idx in arg_indices]
        vars = [(f"?M{i}", arg) for i, arg in enumerate(ground_args)]

        # Construct plan skeleton and predicates
        plan_skeleton = substitute_vars(vars, lifted_task["plan_skeleton"])
        propositions = substitute_vars(vars, lifted_task["predicates"])
        print(f"Plan skeleton: {plan_skeleton}")
        print(f"Predicates: {predicates}")
        input("\nContinue?")
        task_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        "-o",
        type=str,
        nargs="*",
        help="Objects to include for HookReach tasks",
    )
    parser.add_argument(
        "--object-ids",
        "-o-id",
        type=int,
        nargs="*",
        help="Objects IDs to include for HookReach tasks",
    )
    parser.add_argument(
        "--poslimit",
        "-p",
        type=bool,
        default=False,
        help="Apply PosLimit(Predicate) to the rack.",
    )
    args = parser.parse_args()

    LIFTED_TASKS = [
        constrained_packing_task(1, 2, True),
    ]

    objects = parse_dict(OBJECTS, keys=args.objects, values=args.object_ids)

    main(objects, LIFTED_TASKS)
