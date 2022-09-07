from typing import Dict, List, Tuple, Any

import random
from symbolic import parse_proposition

from temporal_policies.envs.pybullet.table.objects import OBJECT_HIERARCHY
from temporal_policies.envs.pybullet.table.predicates import (
    PREDICATE_HIERARCHY,
    UNARY_PREDICATES,
    BINARY_PREDICATES,
)


OBJECTS = ["blue_box", "yellow_box", "red_box", "cyan_box", "purple_box", "hook"]
LOCATIONS = ["table", "rack", "bin"]
LIFTED_OBJECTS = ["?M0", "?M1", "?M2", "?M3", "?M4", "?M5"]
LIFTED_LOCATIONS = ["?L0", "?L1", "?L2"]


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


def lambda_object(x):
    for i, obj_type in enumerate(OBJECT_HIERARCHY):
        if obj_type in x:
            return i


def object_indexer(proposition: str):
    predicate, arguments = parse_proposition(proposition)
    if predicate in UNARY_PREDICATES:
        arg1, arg2 = arguments[0], None
    elif predicate in BINARY_PREDICATES:
        arg1, arg2 = arguments
    else:
        raise ValueError(f"Predicate of type {proposition} is not supported")

    score = 0
    for i, object_type in enumerate(OBJECT_HIERARCHY):
        if object_type in arg1:
            score += i + 1
        if arg2 is not None and object_type in arg2:
            score += i + 1

    return score if score != 0 else 2 * len(OBJECT_HIERARCHY)


def sort_propositions(propositions: List[str]) -> List[str]:
    sorted_propositions = []
    for pred in PREDICATE_HIERARCHY:
        predicates = []
        for prop in propositions:
            if pred == prop.split("(")[0]:
                predicates.append(prop)
        if not predicates:
            continue

        sorted_predicates = sorted(predicates, key=object_indexer)
        sorted_propositions.extend(sorted_predicates)

    return sorted_propositions
