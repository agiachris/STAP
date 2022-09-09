from typing import Any, List, Dict

import argparse

from utils import (
    OBJECTS,
    LIFTED_OBJECTS,
    generate_ground_tasks,
)


def rearrangement_push_task(
    num_tasks: int,
    objects: List[str],
    num_arg_objects: int,
    num_non_arg_objects: int = 0,
    arg_object_zone: str = "inoperationalzone",
    non_arg_object_zone: str = "inobstructionzone",
    poslimit_rack: bool = False,
    aligned_rack: bool = False,
) -> Dict[str, Any]:
    """Generate a rearrangement push lifted task.

    Args:
        num_tasks: Number of ground task instances to sample.
        objects: List of (optionally grounded) box names for the task.
        num_arg_objects: Number of task-relevant boxes to be pulled.
        num_non_arg_objects: Number of task-irrelevant boxes in the environment.
        arg_object_zone: TableBounds predicate arg-object initial zone.
        non_arg_object_zone: TableBounds predicate for non-arg-object initial zone.
        poslimit_rack: Whether or not the rack is at fixed locations.
        aligned_rack: Whether or not the rack frame aligns with the world frame.

    Returns:
        lifted_task: Plan skeleton and initial state of the task
                     with additional metadata to guide sampling.
    """
    assert num_arg_objects >= 1
    assert num_arg_objects + num_non_arg_objects <= len(objects)

    # Base task predicates
    predicates = [
        "free(hook)",
        "inworkspace(hook)",
        "beyondworkspace(rack)",
        "on(rack, table)",
        "on(hook, table)",
    ]
    if poslimit_rack:
        predicates.append("poslimit(rack)")
    if aligned_rack:
        predicates.append("aligned(rack)")

    # Construct plan skeleton and non-arg-object predicates
    plan_skeleton = []
    for non_arg_idx in range(num_non_arg_objects):
        non_arg_object = objects[non_arg_idx]
        plan_skeleton.extend(
            [
                f"pick({non_arg_object}, table)",
                f"place({non_arg_object}, table)",
            ]
        )
        predicates.extend(
            [
                f"free({non_arg_object})",
                f"infront({non_arg_object}, rack)",
                f"on({non_arg_object}, table)",
                f"{non_arg_object_zone}({non_arg_object})",
            ]
        )

    # Construct plan skeleton and arg-object predicates
    for arg_idx in range(num_arg_objects):
        if arg_idx == 0:
            plan_skeleton.append(f"pick(hook, table)")
        arg_object = objects[num_non_arg_objects + arg_idx]
        plan_skeleton.append(f"push({arg_object}, hook, rack)")
        predicates.extend(
            [
                f"free({arg_object})",
                f"infront({arg_object}, rack)",
                f"on({arg_object}, table)",
                f"{arg_object_zone}({arg_object})",
            ]
        )

    lifted_task = {
        "num_tasks": num_tasks,
        "plan_skeleton": plan_skeleton,
        "predicates": predicates,
        "metadata": {"num_lifted_objects": num_arg_objects + num_non_arg_objects},
    }
    return lifted_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        "-o",
        type=str,
        nargs="*",
        help="Names of ground objects for the task suite.",
    )
    args = parser.parse_args()
    assert all(x in OBJECTS for x in args.objects)

    lifted_objects = LIFTED_OBJECTS[: len(args.objects)]
    lifted_tasks = [
        rearrangement_push_task(
            num_tasks=1,
            objects=lifted_objects,
            num_arg_objects=1,
            num_non_arg_objects=1,
            poslimit_rack=True,
            aligned_rack=True,
        ),
        rearrangement_push_task(
            num_tasks=2,
            objects=lifted_objects,
            num_arg_objects=1,
            num_non_arg_objects=2,
            poslimit_rack=True,
            aligned_rack=True,
        ),
        rearrangement_push_task(
            num_tasks=2,
            objects=lifted_objects,
            num_arg_objects=1,
            num_non_arg_objects=3,
            poslimit_rack=True,
            aligned_rack=True,
        ),
    ]

    generate_ground_tasks(args.objects, lifted_tasks)
