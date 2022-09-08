from typing import Any, List, Dict, Tuple

import yaml
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


def constrained_packing_task_phase(
    phase: int,
    objects: List[str],
) -> Tuple[List[str], List[str]]:
    """Generate phase of the constrained packing task.

    Args:
        phase: index of the task phase.
        objects: List of (optionally grounded) box names for the task.

    Returns:
        phase_skeleton: Plan skeleton of this phase.
        phase_predicates: Predicates implied by the phase skeleton.
    """
    arg_object = objects[phase]
    phase_skeleton = [f"pick({arg_object}, table)", f"place({arg_object}, rack)"]
    phase_predicates = [
        f"free({arg_object})",
        f"inworkspace({arg_object})",
        f"on({arg_object}, table)",
    ]
    return phase_skeleton, phase_predicates


def constrained_packing_task(
    num_tasks: int,
    objects: List[str],
    locations: List[str],
    num_arg_objects: int,
    num_non_arg_objects: int = 0,
    poslimit_rack: bool = False,
    aligned_rack: bool = False,
) -> Dict[str, Any]:
    """Generate a constrained packing lifted task.

    Args:
        num_tasks: Number of ground task instances to sample.
        objects: List of (optionally grounded) box names for the task.
        locations: Valid placement locations for the objects.
        num_arg_objects: Number of task-relevant boxes to be pulled.
        num_non_arg_objects: Number of task-irrelevant boxes in the environment.
        poslimit_rack: Whether or not the rack is at fixed locations.

    Returns:
        lifted_task: Plan skeleton and initial state of the task
                     with additional metadata to guide sampling.
    """
    assert num_arg_objects >= 1
    assert num_arg_objects + num_non_arg_objects <= len(objects)

    # Base task predicates
    predicates = ["on(rack, table)", "inworkspace(rack)"]
    if poslimit_rack:
        predicates.append("poslimit(rack)")
    if aligned_rack:
        predicates.append("aligned(rack)")

    # Construct plan skeleton and arg-object predicates
    plan_skeleton = []
    for arg_idx in range(num_arg_objects):
        phase_skeleton, phase_predicates = constrained_packing_task_phase(
            phase=arg_idx,
            objects=objects,
        )
        plan_skeleton.extend(phase_skeleton)
        predicates.extend(phase_predicates)

    # Assign non-arg-object On predicates
    for non_arg_idx in range(num_non_arg_objects):
        non_arg_object = objects[num_arg_objects + non_arg_idx]
        non_arg_location = locations[random.randint(0, len(locations) - 1)]
        predicates.append(f"on({non_arg_object}, {non_arg_location})")

    lifted_task = {
        "num_tasks": num_tasks,
        "plan_skeleton": plan_skeleton,
        "predicates": predicates,
        "metadata": {"num_lifted_objects": num_arg_objects + num_non_arg_objects},
    }
    return lifted_task


def main(objects: Dict[int, str], lifted_tasks: List[Dict[str, Any]]) -> None:
    """Randomly generate an even distribution of lifted evaluation tasks."""
    num_tasks = sum(task["num_tasks"] for task in lifted_tasks)
    for task_idx in range(num_tasks):
        print(f"\nSampling lifted task: {task_idx}")

        # Sample ground objects for constrained packing problem
        lifted_task = get_lifted_task(task_idx, lifted_tasks)
        num_lifted_objects = lifted_task["metadata"]["num_lifted_objects"]
        object_idxs = get_random_indices(objects)[:num_lifted_objects]
        vars = [(f"?M{i}", objects[idx]) for i, idx in enumerate(object_idxs)]

        # Construct action skeleton and initial state
        action_skeleton = substitute_vars(vars, lifted_task["plan_skeleton"])
        propositions = substitute_vars(vars, lifted_task["predicates"])
        initial_state = sort_propositions(propositions)
        task_config = {
            "action_skeleton": action_skeleton,
            "initial_state": initial_state,
        }
        print(yaml.dump(task_config, default_flow_style=False, sort_keys=False))
        input("Continue?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        "-o",
        type=str,
        nargs="*",
        help="Names of ground objects for the task suite.",
    )
    parser.add_argument(
        "--locations",
        "-l",
        type=str,
        nargs="*",
        help="Names of ground placement locations for the task suite.",
    )
    args = parser.parse_args()
    assert all(x in OBJECTS for x in args.objects) and all(
        x in LOCATIONS for x in args.locations
    )

    lifted_objects = LIFTED_OBJECTS[: len(args.objects)]
    lifted_tasks = [
        constrained_packing_task(
            num_tasks=1,
            objects=lifted_objects,
            locations=args.locations,
            num_arg_objects=2,
            num_non_arg_objects=1,
            poslimit_rack=True,
            aligned_rack=True,
        ),
        constrained_packing_task(
            num_tasks=1,
            objects=lifted_objects,
            locations=args.locations,
            num_arg_objects=2,
            num_non_arg_objects=2,
            poslimit_rack=True,
            aligned_rack=True,
        ),
        constrained_packing_task(
            num_tasks=1,
            objects=lifted_objects,
            locations=args.locations,
            num_arg_objects=3,
            num_non_arg_objects=0,
            poslimit_rack=True,
            aligned_rack=True,
        ),
        constrained_packing_task(
            num_tasks=1,
            objects=lifted_objects,
            locations=args.locations,
            num_arg_objects=3,
            num_non_arg_objects=1,
            poslimit_rack=True,
            aligned_rack=True,
        ),
        constrained_packing_task(
            num_tasks=1,
            objects=lifted_objects,
            locations=args.locations,
            num_arg_objects=4,
            num_non_arg_objects=0,
            poslimit_rack=True,
            aligned_rack=True,
        ),
    ]

    main(args.objects, lifted_tasks)
