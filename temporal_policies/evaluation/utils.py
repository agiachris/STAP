import itertools
import pathlib
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np

import symbolic
from temporal_policies import envs, planners
from temporal_policies.envs.pybullet.table import (
    object_state,
    predicates,
    primitives as table_primitives,
)
from temporal_policies.envs.pybullet.table.objects import CLS_TO_PROP_TEST_CLS, Object
from temporal_policies.task_planners.goals import is_valid_goal_props


def seed_generator(
    num_eval: int,
    path_results: Optional[Union[str, pathlib.Path]] = None,
) -> Generator[
    Tuple[
        Optional[int],
        Optional[Tuple[np.ndarray, planners.PlanningResult, Optional[List[float]]]],
    ],
    None,
    None,
]:
    """
    Generates seeds for deterministic evaluation for pybullet.TableEnv.
    """
    if path_results is not None:
        npz_files = sorted(
            pathlib.Path(path_results).glob("results_*.npz"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        for npz_file in npz_files:
            with open(npz_file, "rb") as f:
                npz = np.load(f, allow_pickle=True)
                seed: int = npz["seed"].item()
                rewards = np.array(npz["rewards"])
                plan = planners.PlanningResult(
                    actions=np.array(npz["actions"]),
                    states=np.array(npz["states"]),
                    p_success=npz["p_success"].item(),
                    values=np.array(npz["values"]),
                )
                t_planner: List[float] = npz["t_planner"].item()

            yield seed, (rewards, plan, t_planner)

    if num_eval is not None:
        yield 0, None
        for _ in range(num_eval - 1):
            yield None, None


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
            try:
                task_plans_instantiated.append(
                    [
                        env.get_primitive_info(action_call=action_call)
                        for action_call in task_plan
                    ]
                )
            except Exception as e:
                print(f"Exception: {e}")
                continue
    return task_plans_instantiated


def instantiate_task_plan_primitives(
    task_plan: List[str], env: envs.Env
) -> List[envs.Primitive]:
    """Converts a task plan (a list of action calls) to a list of primitives.

    task_plan has the form:
    ["pick(box, table)", "place(box, table)"]
    """
    return [
        env.get_primitive_info(action_call=action_call) for action_call in task_plan
    ]


def get_callable_goal_props(
    predicted_goal_props_list: List[List[str]], possible_props: List[predicates.Predicate]
) -> List[List[predicates.Predicate]]:
    if not is_valid_goal_props(predicted_goal_props_list, possible_props):
        raise ValueError("Invalid goal props")

    callable_goal_props: List[List[predicates.Predicate]] = []
    for predicted_goal_props in predicted_goal_props_list:
        parsed_goal_props = [
            symbolic.problem.parse_proposition(prop) for prop in predicted_goal_props
        ]
        callable_goal_props.append(get_goal_props_instantiated(parsed_goal_props))
    return callable_goal_props


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
    props_list: predicates.Predicate,
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
        objects[obj_name].enable_custom_pose()
        objects[obj_name].set_custom_pose(obj_state.pose())
    # sim = True to use custom pose
    success = any(
        all([prop.value_simple(objects, sim=True) for prop in props])
        for props in props_list
    )

    return success


# TODO(klin) unclear if this successfully handles the case proptestobjects were handling
# Note: it doesn't: TODO(klin)
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

    # sim = True to use custom pose --- need to update the sim=True code
    initial_object_relationships = [
        str(prop) for prop in possible_props if prop.value_simple(objects, sim=True)
    ]

    # disable custom pose
    for obj_name, obs in zip(objects, observation):
        obj_state = object_state.ObjectState(obs)
        objects[obj_name].disable_custom_pose()

    return initial_object_relationships
