from typing import Any, Dict, List, Literal, Optional, Set, Union, Tuple

import os
import ast
import re
import yaml
import shutil
import itertools
from string import Template
from collections import defaultdict

import tyro
import symbolic

from configs.base_config import PDDLConfig, PolicyDatasetGenerationConfig
from scripts.eval.task_gen import utils
from scripts.train import train_policy


MOVABLE_TYPES = {"box", "tool", "movable"}
UNMOVABLE_TYPES = {"receptacle", "unmovable"}


def to_template_strings(function_call_strs: List[str]):
    def convert_to_template_string(func_call):
        # Extract the function name and the arguments.
        match = re.match(r"(\w+)\((.*)\)", func_call)
        func_name = match.group(1)
        args = match.group(2).split(",")

        # Add a '$' symbol before each argument.
        template_args = [f"${arg.strip()}" for arg in args]

        # Join the function name and the template arguments to create the template string.
        template_string = f"{func_name}({', '.join(template_args)})"

        # Special case 1: if function name is "push", then ONLY add a '$' symbol before the object.
        if func_name == "push":
            template_string = "push($obj, hook, rack)"

        # Special case 2: if function name is "pull", then ONLY add a '$' symbol before the object.
        if func_name == "pull":
            template_string = "pull($obj, hook)"

        return template_string

    # Convert each function call to a template string.
    return [convert_to_template_string(func_call) for func_call in function_call_strs]


def get_placeholders(template_string: str) -> List[str]:
    """
    Get the placeholders in a template string, assuming that the placeholders
    have the form 'action($arg1, $arg2, ...)'.
    """
    placeholders = re.findall(r"\$(\w+)", template_string)
    return placeholders


def generate_pddl_problem(
    problem_name: str,
    pddl_config: PDDLConfig,
    object_types: Dict[str, str],
    symbolic_state: List[str],
    save: bool = True,
) -> symbolic.Problem:
    """Generate a PDDL problem file given a symbolic predicate state.

    Arguments:
        problem_name: PDDL problem name.
        pddl_cgf: PDDLConfig.
        object_types: Dictionary of object names to their type.
        symbolic_state: List of symbolic predicates
        save: Saves PDDL problem file to disk if set True.

    Returns:
        problem: PDDL problem.
    """
    problem: symbolic.Problem = symbolic.Problem("tmp", "workspace")
    for obj, obj_type in object_types.items():
        if obj == "table":
            continue
        problem.add_object(obj, obj_type)
    for prop in symbolic_state:
        problem.add_initial_prop(prop)

    pddl_problem_file = pddl_config.get_problem_file(problem_name)
    if save:
        with open(pddl_problem_file, "w") as f:
            f.write(str(problem))

    return problem


def num_inhand(state: Union[List[str], Set[str]]) -> int:
    """Count the number of objects in the gripper."""
    count = 0
    for predicate in state:
        if "inhand" in predicate:
            count += 1
    return count


def is_hook_on_rack(state: Union[List[str], Set[str]]) -> bool:
    """Return True if state has a hook on a rack."""
    for predicate in state:
        if predicate == "on(hook, rack)":
            return True
    return False


def generate_symbolic_states(
    object_types: Dict[str, str],
    rack_properties: Set[str] = {"aligned", "poslimit"},
    hook_on_rack: bool = True,
) -> List[List[str]]:
    """Generate all possible symbolic states over specified objects.

    Arguments:
        object_types: Dictionary of object names to their type.

    Returns:
        symbolic_states: List of valid symbolic states.
    """
    movable_objects = [
        obj for obj, obj_type in object_types.items() if obj_type in MOVABLE_TYPES
    ]
    unmovable_objects = [
        obj for obj, obj_type in object_types.items() if obj_type in UNMOVABLE_TYPES
    ]
    locations = ["nonexistent($movable)", "inhand($movable)"] + [
        f"on($movable, {obj})" for obj in unmovable_objects
    ]

    # Store possible locations of objects.
    object_locations: Dict[str, List[Set[str]]] = defaultdict(list)
    for obj in movable_objects:
        for loc in locations:
            object_locations[obj].append({Template(loc).substitute(movable=obj)})

    # Rack predicates.
    if "rack" in unmovable_objects:
        rack_predicates = {f"{p}(rack)" for p in rack_properties}
        rack_inworkspace = {"on(rack, table)", "inworkspace(rack)"}.union(
            rack_predicates
        )
        rack_beyondworkspace = {"on(rack, table)", "beyondworkspace(rack)"}.union(
            rack_predicates
        )
        object_locations["rack"].extend([rack_inworkspace, rack_beyondworkspace])

    symbolic_states: List[List[str]] = []
    for state in itertools.product(*object_locations.values()):
        state = set.union(*state)
        if num_inhand(state) > 1 or (not hook_on_rack and is_hook_on_rack(state)):
            continue

        # Filter out nonexistent predicates.
        state = [p for p in state if "nonexistent" not in p]
        symbolic_states.append(utils.sort_propositions(state))

    return symbolic_states


def get_syntactically_valid_actions(
    pddl: symbolic.Pddl,
    object_names: List[str],
) -> List[str]:
    """Get all syntactically valid actions.

    Arguments:
        pddl: PDDL object.
        object_names: List of object names.

    Returns:
        syntactically_valid_actions: List of syntactically valid actions.
    """
    syntactically_valid_actions: List[str] = []
    actions: List[str] = [str(action) for action in pddl.actions]
    action_template_strings = to_template_strings(actions)

    for action_template_string in action_template_strings:
        placeholders = get_placeholders(action_template_string)
        for object_name_combination in itertools.product(
            object_names, repeat=len(placeholders)
        ):
            # Create dictionary mapping placeholders to object names.
            placeholder_to_object_name = dict(
                zip(placeholders, object_name_combination)
            )

            # Create a template string.
            template = Template(action_template_string)

            # Substitute placeholders with object names.
            action = template.substitute(placeholder_to_object_name)
            syntactically_valid_actions.append(action)

    return syntactically_valid_actions


def get_symbolic_actions(
    state: List[str],
    object_types: Dict[str, str],
    pddl_config: PDDLConfig,
) -> List[str]:
    """Compute symbolically valid actions in a given state."""
    problem_name = str(state)
    _ = generate_pddl_problem(
        problem_name=problem_name,
        pddl_config=pddl_config,
        object_types=object_types,
        symbolic_state=state,
        save=True,
    )
    pddl = symbolic.Pddl(
        pddl_config.pddl_domain_file,
        pddl_config.get_problem_file(problem_name),
    )
    actions = pddl.list_valid_actions(pddl.initial_state)
    return actions


def get_state_object_types(
    state: List[str], object_types: Dict[str, str]
) -> Dict[str, str]:
    """Return dictionary of objects to object types for objects in state."""
    state_objects = set()
    for prop in state:
        state_objects = state_objects.union(set(symbolic.parse_args(prop)))
    return {
        obj: obj_type for obj, obj_type in object_types.items() if obj in state_objects
    }


def get_states_to_primitives(
    states_to_actions: Dict[str, List[str]], primitive: str
) -> Dict[str, List[str]]:
    """Get mapping from states to specified primitive actions."""
    states_to_primitives: Dict[str, List[str]] = {}
    for state, actions in states_to_actions.items():
        states_to_primitives[state] = [a for a in actions if primitive in a]
    return states_to_primitives


def get_env_config(
    states_to_primitives: Dict[str, List[str]],
    primitive: str,
    template_yaml_path: str,
    gui: bool = False,
    seed: int = 0,
    symbolic_action_type: Literal["valid", "invalid"] = "valid",
    save_env_config: bool = True,
    env_config_path: Optional[str] = None,
    env_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct primitive-specific pybullet environment yaml."""
    with open(template_yaml_path, "r") as f:
        env_config = yaml.safe_load(f)

    tasks = []
    for initial_state, primitive_list in states_to_primitives.items():
        tasks.extend(
            [
                {
                    "initial_state": ast.literal_eval(initial_state),
                    "action_skeleton": [p],
                }
                for p in primitive_list
                if primitive in p
            ]
        )

    if symbolic_action_type == "valid":
        # Ensure probabilities of pick(hook) and pick(box) are equal.
        if primitive == "pick":
            num_pick_hook_actions = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    num_pick_hook_actions += 1
            num_pick_box_actions = len(tasks) - num_pick_hook_actions

            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    task["prob"] = 0.5 * (1 / num_pick_hook_actions)
                else:
                    task["prob"] = 0.5 * (1 / num_pick_box_actions)

        # Ensure probabilities of place(hook) and place(box) are equal.
        if primitive == "place":
            num_place_hook_actions = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    num_place_hook_actions += 1
            num_place_box_actions = len(tasks) - num_place_hook_actions

            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    task["prob"] = 0.5 * (1 / num_place_hook_actions)
                else:
                    task["prob"] = 0.5 * (1 / num_place_box_actions)

    elif symbolic_action_type == "invalid":
        pass

    else:
        raise ValueError(f"Support for {symbolic_action_type} not implemented.")

    env_config["env_kwargs"]["tasks"] = tasks
    env_config["env_kwargs"]["gui"] = gui
    env_config["env_kwargs"]["name"] = (
        f"{primitive}_{seed}" if env_name is None else env_name
    )
    env_config["env_kwargs"]["primitives"] = [primitive]

    if save_env_config:
        if env_config_path is not None:
            if os.path.exists(env_config_path):
                raise ValueError(f"File {env_config_path} already exists.")
            with open(env_config_path, "w") as f:
                yaml.safe_dump(env_config, f)
        else:
            raise ValueError("Require environment configuration save path.")

    return env_config


def main(config: PolicyDatasetGenerationConfig):
    """Create a primitive specific dataset of (s, a, r, s') transitions.

    Arguments:
        config: PolicyDatasetGenerationConfig.
    """
    # Create environment root directory.
    if not os.path.exists(config.env_root_dir):
        os.makedirs(config.env_root_dir, exist_ok=True)

    # Create temporary PDDL problem subdirectory.
    config.pddl_config.problem_subdir = f"{config.env_name}"
    pddl_problem_dir = config.pddl_config.pddl_problem_dir
    if not os.path.exists(pddl_problem_dir):
        os.makedirs(pddl_problem_dir)

    # Compute symbolically valid and invalid actions.
    states_to_actions: Dict[str, List[str]] = defaultdict(list)
    for state in generate_symbolic_states(config.object_types):
        state_object_types = get_state_object_types(state, config.object_types)
        actions = get_symbolic_actions(state, state_object_types, config.pddl_config)

        if len(actions) > 0:
            states_to_actions[str(state)] = actions

    # Delete temporary PDDL problem subdirectory.
    shutil.rmtree(pddl_problem_dir)

    states_to_primitives: Dict[str, List[str]] = get_states_to_primitives(
        states_to_actions=states_to_actions,
        primitive=config.primitive,
    )
    num_states = len(states_to_primitives.keys())
    num_primitives = sum(len(p) for p in states_to_primitives.values())
    avg_primitives = float(num_primitives) / float(num_states)

    # Save record of states and primitives.
    env_name = os.path.splitext(config.env_config_path)[0]
    with open(f"{env_name}_states_actions.txt", "w") as f:
        f.write(f"States: {num_states}\n")
        f.write(f"Primitives: {num_primitives}\n")
        f.write(f"Primitives / State: {avg_primitives}\n\n")
        for state, primitives in states_to_primitives.items():
            f.write(f"State: {state}\n")
            f.write(f"Action: {sorted(primitives)}\n\n")

    env_config = get_env_config(
        states_to_primitives=states_to_primitives,
        template_yaml_path=config.template_env_yaml,
        primitive=config.primitive,
        seed=config.seed,
        symbolic_action_type=config.symbolic_action_type,
        save_env_config=config.save_env_config,
        env_config_path=config.env_config_path,
        env_name=config.env_name,
    )

    train_policy.train(
        config.path,
        trainer_config=config.trainer_config,
        agent_config=config.agent_config,
        env_config=env_config,
        device=config.device,
        seed=config.seed,
    )


if __name__ == "__main__":
    tyro.cli(main)
