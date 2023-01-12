"""
Generates a dataset of (s, a, s', r) pairs for the TableEnv gym environment.

Usage:

PYTHONPATH=. python scripts/data/generate_policy_dataset.py 
--config.pddl-cfg.pddl-domain template --config.num-train-steps 0 --config.num-eval-episodes 0
--config.exp-name 20230108/dataset_collection/  --config.primitive pick
--config.min-num-box-obj 1 --config.max-num-box-obj 4

N.B. last two configs to avoid training a policy and/or evaluating the policy.
Intended usage is to generate a dataset of (s, a, s', r) pairs with multiple processes
and then train a Q function and policy on the dataset.
"""

from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import ast
import math
import re
import yaml
import itertools
from string import Template
from collections import defaultdict

import tyro
import symbolic

from configs.base_config import PDDLConfig, PolicyDatasetGenerationConfig
from scripts.eval.task_gen import utils
from scripts.train import train_policy


MOVABLE_TYPES = {
    "box",
    "tool"
    "movable"
}


UNMOVABLE_TYPES = {
    "receptacle"
    "unmovable"
}


def to_template_strings(function_call_strs: List[str]):
    def convert_to_template_string(func_call):
        # Extract the function name and the arguments
        match = re.match(r"(\w+)\((.*)\)", func_call)
        func_name = match.group(1)
        args = match.group(2).split(",")

        # Add a '$' symbol before each argument
        template_args = [f"${arg.strip()}" for arg in args]

        # Join the function name and the template arguments to create the template string
        template_string = f"{func_name}({', '.join(template_args)})"

        # special case 1: if function name is "push", then ONLY add a '$' symbol before the object
        if func_name == "push":
            template_string = "push($obj, hook, rack)"

        # special case 2: if function name is "pull", then ONLY add a '$' symbol before the object
        if func_name == "pull":
            template_string = "pull($obj, hook)"

        return template_string

    # Convert each function call to a template string
    template_strings = [
        convert_to_template_string(func_call) for func_call in function_call_strs
    ]
    return template_strings


def get_placeholders(template_string: str) -> List[str]:
    """
    Get the placeholders in a template string, assuming that the placeholders
    have the form 'action($arg1, $arg2, ...)'.
    """
    placeholders = re.findall(r"\$(\w+)", template_string)
    return placeholders


def get_syntactically_valid_actions(
    pddl: symbolic.Pddl,
    object_names: List[str],
) -> List[str]:
    """Get all syntactically valid actions.

    Args:
        pddl (symbolic.PDDL): PDDL object.
        object_names (List[str]): List of object names.

    Returns:
        List[str]: List of syntactically valid actions.
    """
    # get all syntactically valid actions
    syntactically_valid_actions: List[str] = []
    actions: List[str] = [str(action) for action in pddl.actions]
    action_template_strings = to_template_strings(actions)
    for action_template_string in action_template_strings:
        placeholders = get_placeholders(action_template_string)
        # get all possible combinations of object names
        object_name_combinations = list(
            itertools.product(object_names, repeat=len(placeholders))
        )
        for object_name_combination in object_name_combinations:
            # create a dictionary mapping placeholders to object names
            placeholder_to_object_name = dict(
                zip(placeholders, object_name_combination)
            )
            # create a template string
            template = Template(action_template_string)
            # substitute placeholders with object names
            action = template.substitute(placeholder_to_object_name)
            syntactically_valid_actions.append(action)

    return syntactically_valid_actions


def num_inhand(state: Union[List[str], Set[Tuple[str, str]]]) -> int:
    """Count the number of objects in the gripper.

    Args:
        state: List of symbolic predicate states.

    Returns:
        int: Number of objects in the gripper
    """
    num_inhand = 0
    for symbolic_predicate_state in state:
        if "inhand" in symbolic_predicate_state:
            num_inhand += 1
    return num_inhand


def is_hook_on_rack(state: Union[List[str], Set[Tuple[str, str]]]) -> bool:
    """Check if the hook is on the rack.

    Args:
        state: List of symbolic predicate states.

    Returns:
        bool: True if the hook is on the rack, False otherwise.
    """
    for symbolic_predicate_state in state:
        if "hook" in symbolic_predicate_state and "rack" in symbolic_predicate_state:
            return True
    return False


def generate_symbolic_states(
    object_types: Dict[str, str],
    rack_properties: Set[str] = {"aligned", "poselimit"},
    hook_on_rack: bool = True,
) -> List[Set[str]]:
    """Generate all possible symbolic states over specified objects.
    
    Arguments:
        object_types: Dictionary of object names to their type.

    Returns:
        symbolic_states: List of valid symbolic states.
    """
    movable_objects = [obj for obj, obj_type in object_types if obj_type in MOVABLE_TYPES]
    unmovable_objects = [obj for obj, obj_type in object_types if obj_type in UNMOVABLE_TYPES]
    locations = ["nonexistent($movable)", "inhand($movable)"] + [f"on($movable, {obj})" for obj in unmovable_objects]

    # Store possible locations of objects.
    object_locations: Dict[str, List[Set[str]]] = defaultdict(list)
    for obj in movable_objects:
        for loc in locations:
            object_locations[obj].append({Template(loc).substitute(movable=obj)})

    # Rack predicates.
    if "rack" in unmovable_objects:
        rack_predicates = {f"{p}(rack)" for p in rack_properties}
        rack_inworkspace = {"on(rack, table)", "inworkspace(rack)"}.union(rack_predicates)
        rack_beyondworkspace = {"on(rack, table)", "beyondworkspace(rack)"}.union(rack_predicates)
        object_locations["rack"].extend([rack_inworkspace, rack_beyondworkspace])


    breakpoint()

    candidate_states = list(itertools.product(*object_locations.values()))
    symbolic_states = []
    for state in candidate_states:
        breakpoint()
        state = set.union(*state)

        if num_inhand(state) > 1 or (not hook_on_rack and is_hook_on_rack(state)):
            continue

        # Filter out nonexistent predicates.
        state = {p for p in state if "nonexistent" not in p}
        symbolic_states.append(state)

    return symbolic_states
 

def generate_pddl_problem_from_state_set(
    problem_name: str,
    pddl_cfg: PDDLConfig,
    objects_with_properties: List[Tuple[str, str]],
    symbolic_predicate_state: Set[str],
    save_pddl_file: bool = True,
) -> symbolic.Problem:
    """Generate a PDDL problem file given a symbolic predicate state.

    Args:
        symbolic_predicate_state (Set[str]): Set of symbolic predicate states.
        E.g. {on(red_box, table), on(blue_box, rack), ...}
        save_pddl_file: Whether to save the PDDL problem file to disk. symbolic
            only seems to read from a file path saved to disk.
    """
    problem: symbolic.Problem = symbolic.Problem("test-prob", "workspace")
    # add objects to the problem for loop
    for obj, obj_property in objects_with_properties:
        if obj != "table":
            problem.add_object(obj, obj_property)

    for prop in symbolic_predicate_state:
        problem.add_initial_prop(prop)

    problem_pddl_file = pddl_cfg.get_problem_file(problem_name)
    if save_pddl_file:
        with open(problem_pddl_file, "w") as f:
            f.write(str(problem))

    return problem


def find_gcd(numbers):
    gcd = numbers[0]
    for i in range(1, len(numbers)):
        gcd = math.gcd(gcd, numbers[i])
    return gcd


def get_primitive_counts(
    symbolic_states_to_actions: Dict[str, Dict[str, List[str]]],
    primitives: List[str],
    find_count_ratios: bool = False,
) -> Dict[str, int]:
    """Get the number of primitive actions for each primitive.

    Args:
        symbolic_states_to_actions (Dict[str, Dict[str, List[str]]]): Dictionary
        mapping symbolic states to actions.

    Returns:
        Dict[str, int]: Dictionary mapping primitive to number of primitive
        actions.
    """
    primitive_counts = defaultdict(int)
    # get total counts that each primitive is used from valid_states_to_actions
    for _, symbolic_action_dict in symbolic_states_to_actions.items():
        for symbolic_action in symbolic_action_dict["symbolically_valid_actions"]:
            for primitive in primitives:
                if primitive in symbolic_action:
                    primitive_counts[primitive] += 1

    if find_count_ratios:
        # find highest common denominator for the counts
        gcd = find_gcd(list(primitive_counts.values()))

        # divide each count by the gcd
        for primitive in primitives:
            primitive_counts[primitive] //= gcd
    return primitive_counts


def get_env_config(
    symbolic_states_to_actions: Dict[str, Dict[str, List[str]]],
    template_yaml_path: str = "",
    env_primitive: Literal["pick", "place", "pull", "push"] = "pick",
    gui: bool = False,
    seed: int = 0,
    symbolic_action_type: Literal[
        "symbolically_valid_actions", "syntactically_valid_symbolically_invalid_actions"
    ] = "symbolically_valid_actions",
    save_primitive_env_config: bool = False,
    save_primitive_env_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    1. Load a default environment from a yaml file.
    2. Create task distribution of initial states and actions
    3. Only include actions that match env_primitive
    """
    with open(template_yaml_path, "r") as f:
        env_config = yaml.safe_load(f)

    tasks = []
    for symbolic_state, symbolic_action_dict in symbolic_states_to_actions.items():
        for symbolic_action in symbolic_action_dict[symbolic_action_type]:
            if env_primitive in symbolic_action:
                tasks.append(
                    {
                        "initial_state": ast.literal_eval(symbolic_state),
                        "action_skeleton": [symbolic_action],
                    }
                )

    if symbolic_action_type == "symbolically_valid_actions":
        # ensure probabilities of pick(hook) and pick(box) are equal
        if env_primitive == "pick":
            num_pick_hook_actions = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    num_pick_hook_actions += 1
            num_pick_box_actions = len(tasks) - num_pick_hook_actions
            # set probabilities associated with pick hook to 0.5
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    task["prob"] = 0.5 * (1 / num_pick_hook_actions)
                else:
                    task["prob"] = 0.5 * (1 / num_pick_box_actions)

            sum_prob_pick_hook = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    sum_prob_pick_hook += task["prob"]
            assert math.isclose(
                sum_prob_pick_hook, 0.5, abs_tol=1e-5
            ), "sum of probabilities of pick hook actions should be 0.5"

            sum_prob_pick_box = 0
            for task in tasks:
                if "hook" not in task["action_skeleton"][0]:
                    sum_prob_pick_box += task["prob"]
            assert math.isclose(
                sum_prob_pick_box, 0.5, abs_tol=1e-5
            ), "sum of probabilities of pick box actions should be 0.5"

        # ensure probabilities of place(hook) and place(box) are equal
        if env_primitive == "place":
            num_place_hook_actions = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    num_place_hook_actions += 1
            num_place_box_actions = len(tasks) - num_place_hook_actions
            # set probabilities associated with place hook to 0.5
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    task["prob"] = 0.5 * (1 / num_place_hook_actions)
                else:
                    task["prob"] = 0.5 * (1 / num_place_box_actions)

    env_config["env_kwargs"]["tasks"] = tasks
    env_config["env_kwargs"]["gui"] = gui
    env_config["env_kwargs"]["name"] = f"{env_primitive}_{seed}"
    env_config["env_kwargs"]["primitives"] = [env_primitive]

    if save_primitive_env_config:
        assert save_primitive_env_config_path is not None, (
            "Must specify save_primitive_env_config_path if "
            "save_primitive_env_config is True."
        )
        with open(save_primitive_env_config_path, "w") as f:
            print(f"Saving env config to {save_primitive_env_config_path}")
            yaml.safe_dump(env_config, f)
    return env_config


def main(config: PolicyDatasetGenerationConfig):
    """Create a primitive specific dataset of (s, a, r, s') transitions.

    Arguments:
        config: PolicyDatasetGenerationConfig.
    """
    symbolic_states: List[Set[str]] = generate_symbolic_states(config.object_types)
    

    valid_states_to_actions: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    states, actions = [], []
    count = 0
    for symbolic_predicate_state in symbolic_states:
        problem_name = "val-training" + str(utils.sort_propositions(list(symbolic_predicate_state)))
        problem: symbolic.Problem = generate_pddl_problem_from_state_set(
            problem_name,
            config.pddl_cfg,
            objects_with_properties,
            utils.sort_propositions(list(symbolic_predicate_state)),
            save_pddl_file=True,
        )
        with open(config.pddl_cfg.get_problem_file(problem_name), "r") as f:
            print(f.read())
            print("read problem file successfully")
        
        pddl = symbolic.Pddl(
            config.pddl_cfg.pddl_domain_file,
            config.pddl_cfg.get_problem_file(problem_name),
        )

        state = pddl.initial_state
        symbolically_valid_actions = pddl.list_valid_actions(state)
        syntactically_valid_actions = get_syntactically_valid_actions(
            pddl,
            [obj_name for obj_name, _ in objects_with_properties],
        )

        # remove symbolically_valid_actions from syntactically_valid_actions
        syntactically_valid_symbolically_invalid_actions = [
            action
            for action in syntactically_valid_actions
            if not action in symbolically_valid_actions
        ]
        state_key = str(utils.sort_propositions(list(symbolic_predicate_state)))

        # check if sort_propositions is needed
        valid_states_to_actions[state_key][
            "symbolically_valid_actions"
        ] = symbolically_valid_actions
        valid_states_to_actions[state_key][
            "syntactically_valid_symbolically_invalid_actions"
        ] = syntactically_valid_symbolically_invalid_actions
        assert len(symbolically_valid_actions) > 0, "No valid actions for this state"
        assert (
            len(syntactically_valid_symbolically_invalid_actions) > 0
        ), "No syntactically valid symbolically_invalid_actions for this state"
        count += len(
            symbolically_valid_actions
        )  # + len(syntactically_valid_symbolically_invalid_actions)
        states.append(utils.sort_propositions(list(symbolic_predicate_state)))
        actions.append(symbolically_valid_actions)

    print(f"valid_symbolic_predicate_states: {len(symbolic_states)}")
    print(f"count: {count}")
    # save these state actions to a file for debugging
    with open("states_actions.txt", "w") as f:
        for state, action in zip(states, actions):
            f.write(f"state: {sorted(list(state))}\n")
            f.write(f"action: {sorted(list(action))}\n\n")

    print(get_primitive_counts(valid_states_to_actions, ["pick"]))
    print(get_primitive_counts(valid_states_to_actions, ["place"]))
    print(get_primitive_counts(valid_states_to_actions, ["pull"]))
    print(get_primitive_counts(valid_states_to_actions, ["push"]))

    env_config = get_env_config(
        valid_states_to_actions,
        template_yaml_path=config.template_yaml_path,
        env_primitive=config.primitive,
        gui=config.gui,
        seed=config.seed,
        symbolic_action_type=config.symbolic_action_type,
        save_primitive_env_config=config.save_primitive_env_config,
        save_primitive_env_config_path=config.save_primitive_env_config_path,
    )

    train_policy.train(
        config.path,
        trainer_config=config.trainer_config,
        agent_config=config.agent_config,
        env_config=env_config,
        eval_env_config=None,  # prevent spinning up eval env
        encoder_checkpoint=config.encoder_checkpoint,
        eval_recording_path=config.eval_recording_path,
        resume=config.resume,
        overwrite=config.overwrite,
        device=config.device,
        seed=config.seed,
        gui=config.gui,
        use_curriculum=config.use_curriculum,
        num_pretrain_steps=config.num_pretrain_steps,
        num_train_steps=config.num_train_steps,
        num_eval_episodes=config.num_eval_episodes,
        num_env_processes=config.num_env_processes,
        num_eval_env_processes=config.num_eval_env_processes,
    )


if __name__ == "__main__":
    tyro.cli(main)
