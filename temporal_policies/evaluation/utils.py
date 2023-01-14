import itertools
from typing import Dict, Generator, List, Tuple
import numpy as np

import symbolic
from temporal_policies import envs
from temporal_policies.envs.pybullet.table import (
    object_state,
    predicates,
    primitives as table_primitives,
)
from temporal_policies.envs.pybullet.table.objects import CLS_TO_PROP_TEST_CLS, Object
from temporal_policies.task_planners.goals import is_valid_goal_props


def get_goal_props_instantiated(
    props: List[Tuple[str, List[str]]]
) -> List[predicates.Predicate]:
    goal_props = []
    for prop in props:
        if prop[0] in predicates.UNARY_PREDICATES:
            predicate_cls = predicates.UNARY_PREDICATES[prop[0]]
            goal_props.append(predicate_cls([prop[1][0]]))
        elif prop[0] in predicates.BINARY_PREDICATES:
            predicate_cls = predicates.BINARY_PREDICATES[prop[0]]
            goal_props.append(predicate_cls([prop[1][0], prop[1][1]]))
    return goal_props


def get_task_plan_primitives_instantiated(
    task_plans: List[List[str]], env: envs.Env
) -> List[List[envs.Primitive]]:
    """Converts a list of task plans (each a list of action calls) to a list of task plans (each a list of primitives).

    task_plans has the form:
    [
        ["pick(box, table)", "place(box, table)"],
        ["pick(hook, table)", "place(hook, table)"],
        ...
    ]
    """
    task_plans_instantiated = []
    for task_plan in task_plans:
        primitives_in_plan = [action_call.split("(")[0] for action_call in task_plan]
        # check that the current task plan uses valid primitives. If not, skip it.
        if not all([primitive in env.primitives for primitive in primitives_in_plan]):
            continue
        # check current task plan is not an empty list
        if len(task_plan) == 0:
            continue
        else:
            task_plans_instantiated.append(
                [
                    env.get_primitive_info(action_call=action_call)
                    for action_call in task_plan
                ]
            )
    return task_plans_instantiated


def get_callable_goal_props(
    predicted_goal_props: List[str], possible_props: List[predicates.Predicate]
) -> List[predicates.Predicate]:
    if not is_valid_goal_props(predicted_goal_props, possible_props):
        raise ValueError("Invalid goal props")

    parsed_goal_props = [
        symbolic.problem.parse_proposition(prop) for prop in predicted_goal_props
    ]
    return get_goal_props_instantiated(parsed_goal_props)


def get_possible_props(
    objects: Dict[str, Object], available_predicates: List[str]
) -> List[predicates.Predicate]:
    """Returns all possible props given the objects in the scene and available predicates."""
    possible_props = []
    for predicate in available_predicates:
        if predicate in predicates.UNARY_PREDICATES:
            predicate_cls = predicates.UNARY_PREDICATES[predicate]
            for obj in objects.values():
                possible_props.append(predicate_cls([obj.name]))
        elif predicate in predicates.BINARY_PREDICATES:
            predicate_cls = predicates.BINARY_PREDICATES[predicate]
            for obj1, obj2 in itertools.permutations(objects.values(), 2):
                possible_props.append(predicate_cls([obj1.name, obj2.name]))
    return possible_props


# def get_possible_props(objects: Dict[str, Object], available_predicates: List[str], unary_predicates: List[str], binary_predicates: List[str]) -> List[predicates.Predicate]:
#     """Returns all possible props given the objects in the scene and available predicates."""
#     possible_props = []
#     for predicate in available_predicates:
#         if predicate in unary_predicates:
#             possible_props += [predicate([obj]) for obj in objects.values()]
#         elif predicate in binary_predicates:
#             # loop through all pairs of objects (don't duplicate)
#             possible_props += [predicate([obj1, obj2]) for obj1, obj2 in itertools.combinations(objects.values(), 2)]
#         else:
#             # handle case where predicate is not a unary or binary predicate
#             pass  # you can add code here to raise an exception or return an empty list
#     return possible_props


def get_prop_testing_objs(env: envs.Env) -> Dict[str, Object]:
    """Returns a dict of objects that can be used to test propositions."""
    prop_testing_objs: Dict[str, Object] = {}
    for obj_name, obj in env.objects.items():
        obj_cls_name = obj.__class__.__name__
        prop_test_obj = CLS_TO_PROP_TEST_CLS[obj_cls_name](obj)
        prop_testing_objs[obj_name] = prop_test_obj
    return prop_testing_objs


def is_satisfy_goal_props(
    props: predicates.Predicate,
    objects: Dict[str, Object],
    state: np.ndarray,
    use_hand_state: bool = False,
) -> bool:
    """Returns True if all props for hold for the given objects, False otherwise.

    Args:
        props: list of predicates to test
        objects: dict of objects in the scene
        state: state (either observation or predicted dynamics state) of the scene
    """
    if not use_hand_state:
        print(
            f"Note: cutting out the first observation entry (ee observation) ---- skipping inhand(a)"
        )
        # cutting out the EE observation means that I probably won't have access to inhand(a)
        state = state[1:]

    for obj_name, s in zip(objects, state):
        obj_state = object_state.ObjectState(s)
        objects[obj_name].set_custom_pose(obj_state.pose())

    success = all([prop.value_simple(objects) for prop in props])

    return success


def get_object_relationships(
    observation: np.ndarray,
    objects: Dict[str, Object],
    available_predicates: List[str],
    use_hand_state: bool = False,
) -> List[str]:
    if not use_hand_state:
        print(
            f"Note: cutting out the first observation entry (ee observation) ---- skipping inhand(a)"
        )
        # cutting out the EE observation means that I probably won't have access to inhand(a)
        observation = observation[1:]
    possible_props = get_possible_props(objects, available_predicates)
    # apply raw observations to objects
    for obj_name, obs in zip(objects, observation):
        obj_state = object_state.ObjectState(obs)
        objects[obj_name].enable_custom_pose()
        objects[obj_name].set_custom_pose(obj_state.pose())

    initial_object_relationships = [
        prop for prop in possible_props if prop.value_simple(objects)
    ]

    # disable custom pose
    for obj_name, obs in zip(objects, observation):
        obj_state = object_state.ObjectState(obs)
        objects[obj_name].disable_custom_pose()

    return initial_object_relationships
