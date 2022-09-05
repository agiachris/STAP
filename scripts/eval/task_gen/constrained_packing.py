from typing import Any, List, Tuple, Dict

import random
import numpy as np

from temporal_policies.envs.pybullet.table.predicates import PREDICATE_HIERARCHY


def get_random_indices(objects: Dict[int, str]) -> List[int]:
    arg_indices = list(range(len(objects)))
    random.shuffle(arg_indices)
    return arg_indices


def get_lifted_task(
    task_idx: int, lifted_tasks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    task_ranges = [
        sum(task["num_tasks"] for task in lifted_tasks[: k + 1])
        for k in range(len(lifted_tasks))
    ]
    for k in range(0, len(task_ranges)):
        if k == 0 and task_idx < task_ranges[0]:
            return lifted_tasks[0]
        elif task_idx >= task_ranges[k - 1] and task_idx < task_ranges[k]:
            return lifted_tasks[k]
    raise ValueError(f"Provided task_idx {task_idx} is invalid.")


def substitute_vars(vars: List[Tuple[str, str]], elements: List[str]) -> List[str]:
    annotated_elements = []
    for e in elements:
        for v in vars:
            if v[0] in e:
                e = e.replace(v[0], v[1])
        annotated_elements.append(e)
    return annotated_elements


def main(objects: Dict[int, str], lifted_tasks: List[Dict[str, Any]]) -> None:
    """Randomly generate an even distribution of lifted evaluation tasks."""
    task_idx = 0
    num_tasks = sum(task["num_tasks"] for task in lifted_tasks)
    while task_idx < num_tasks:
        print(f"\nSampling lifted task: {task_idx + 1}")

        # Sample ground objects for constrained packing problem
        lifted_task = get_lifted_task(task_idx, lifted_tasks)
        num_lifted_args = len(lifted_task["plan_skeleton"]) // 2
        arg_indices = random.sample(get_random_indices(objects), num_lifted_args)
        ground_args = [objects[idx] for idx in arg_indices]
        vars = [(f"?M{i}", arg) for i, arg in enumerate(ground_args)]

        # Construct plan skeleton and predicates
        plan_skeleton = substitute_vars(vars, lifted_task["plan_skeleton"])
        predicates = substitute_vars(vars, lifted_task["predicates"])
        print(f"Plan skeleton: {plan_skeleton}")
        print(f"Predicates: {predicates}")
        input("\nContinue?")
        task_idx += 1


if __name__ == "__main__":
    objects = {
        0: "blue_box",
        1: "yellow_box",
        2: "red_box",
        3: "cyan_box",
        4: "purple_box",
        5: "hook",
    }
    lifted_objects = ["?M0", "?M1", "?M2", "?M3", "?M4", "?M5"]
    lifted_tasks = [
        {
            "num_tasks": 1,
            "plan_skeleton": [
                x
                for p in zip(
                    [f"pick({m}, table)" for m in lifted_objects[:3]],
                    [f"place({m}, rack)" for m in lifted_objects[:3]],
                )
                for x in p
            ],
            "predicates": [f"free({m})" for m in lifted_objects[:3]]
            + ["inworkspace(rack)"]
            + [f"inworkspace({m})" for m in lifted_objects[:3]]
            + ["on(rack, table)"]
            + [f"on({m}, table)" for m in lifted_objects[:3]],
        },
        {
            "num_tasks": 2,
            "plan_skeleton": [
                x
                for p in zip(
                    [f"pick({m}, table)" for m in lifted_objects[:4]],
                    [f"place({m}, rack)" for m in lifted_objects[:4]],
                )
                for x in p
            ],
            "predicates": [f"free({m})" for m in lifted_objects[:4]]
            + ["inworkspace(rack)"]
            + [f"inworkspace({m})" for m in lifted_objects[:4]]
            + ["on(rack, table)"]
            + [f"on({m}, table)" for m in lifted_objects[:4]],
        },
        {
            "num_tasks": 4,
            "plan_skeleton": [
                x
                for p in zip(
                    [f"pick({m}, table)" for m in lifted_objects[:5]],
                    [f"place({m}, rack)" for m in lifted_objects[:5]],
                )
                for x in p
            ],
            "predicates": [f"free({m})" for m in lifted_objects[:5]]
            + ["inworkspace(rack)"]
            + [f"inworkspace({m})" for m in lifted_objects[:5]]
            + ["on(rack, table)"]
            + [f"on({m}, table)" for m in lifted_objects[:5]],
        },
        {
            "num_tasks": 3,
            "plan_skeleton": [
                x
                for p in zip(
                    [f"pick({m}, table)" for m in lifted_objects[:6]],
                    [f"place({m}, rack)" for m in lifted_objects[:6]],
                )
                for x in p
            ],
            "predicates": [f"free({m})" for m in lifted_objects[:6]]
            + ["inworkspace(rack)"]
            + [f"inworkspace({m})" for m in lifted_objects[:6]]
            + ["on(rack, table)"]
            + [f"on({m}, table)" for m in lifted_objects[:6]],
        },
    ]
    main(objects, lifted_tasks)
