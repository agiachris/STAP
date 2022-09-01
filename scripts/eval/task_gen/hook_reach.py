from typing import Any, List, Tuple, Dict

import random
import numpy as np


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


def main(
    boxes: Dict[int, str],
    locations: Dict[int, str],
    lifted_tasks: List[Dict[str, Any]],
) -> None:
    """Randomly generate an even distribution of lifted evaluation tasks."""
    task_idx = 0
    num_tasks = sum(task["num_tasks"] for task in lifted_tasks)
    arg_indices = get_random_indices(boxes)
    num_non_arg_values = get_random_indices(boxes)
    while task_idx < num_tasks:
        while arg_indices:
            print(f"\nSampling lifted task: {task_idx + 1}")

            # Sample unused target box
            target_box_idx = arg_indices.pop()
            target_box = boxes[target_box_idx]

            # Sample number of non-args in the environment
            num_non_args = num_non_arg_values.pop()
            non_arg_indices = [x for x in range(len(boxes)) if x != target_box_idx]
            random.shuffle(non_arg_indices)
            non_args = [boxes[non_arg_indices[x]] for x in range(num_non_args)]

            # Ensure all boxes are unique
            box_indices = np.array([target_box_idx, *non_arg_indices])
            assert len(np.unique(box_indices)) == len(box_indices)

            # Construct plan skeleton
            lifted_task = get_lifted_task(task_idx, lifted_tasks)
            vars = [("?B", target_box)]

            # Construct predicates
            predicates = [
                *lifted_task["predicates"],
                f"beyondworkspace({target_box})",
                f"on({target_box}, table)",
            ]
            for non_arg in non_args:
                location = locations[random.randint(0, len(locations) - 1)]
                predicates.append(f"on({non_arg}, {location})")

            print(
                f"Plan skeleton: {substitute_vars(vars, lifted_task['plan_skeleton'])}"
            )
            print(f"Predicates: {predicates}")
            task_idx += 1

        arg_indices = get_random_indices(boxes)
        num_non_arg_values = get_random_indices(boxes)


if __name__ == "__main__":
    boxes = {
        0: "blue_box",
        1: "yellow_box",
        2: "red_box",
        3: "cyan_box",
        4: "purple_box",
    }
    locations = {
        0: "table",
        1: "rack",
    }
    lifted_tasks = [
        {
            "num_tasks": 4,
            "plan_skeleton": [
                "pick(hook, table)",
                "pull(?B, hook)",
                "place(hook, table)",
                "pick(?B, table)",
            ],
            "predicates": ["inworkspace(hook)", "on(rack, table)", "on(hook, table)"],
        },
        {
            "num_tasks": 2,
            "plan_skeleton": [
                "pick(hook, table)",
                "pull(?B, hook)",
                "place(hook, rack)",
                "pick(?B, table)",
            ],
            "predicates": [
                "inworkspace(rack)",
                "inworkspace(hook)",
                "on(rack, table)",
                "on(hook, table)",
            ],
        },
        {
            "num_tasks": 2,
            "plan_skeleton": [
                "pick(hook, rack)",
                "pull(?B, hook)",
                "place(hook, table)",
                "pick(?B, table)",
            ],
            "predicates": [
                "inworkspace(rack)",
                "inworkspace(hook)",
                "on(rack, table)",
                "on(hook, rack)",
            ],
        },
        {
            "num_tasks": 2,
            "plan_skeleton": [
                "pick(hook, rack)",
                "pull(?B, hook)",
                "place(hook, rack)",
                "pick(?B, table)",
            ],
            "predicates": [
                "inworkspace(rack)",
                "inworkspace(hook)",
                "on(rack, table)",
                "on(hook, rack)",
            ],
        },
    ]
    main(boxes, locations, lifted_tasks)
