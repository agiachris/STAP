import ast
from dataclasses import asdict, dataclass
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union
import openai
import json

import symbolic

from temporal_policies.task_planners.lm_data_structures import CurrentExample, InContextExample, Result

# complication: Episode can be used as a single CoT prompt, or as a prompt for current context
# complication: Human might give an impossible goal ... 

# Single prompt unit that can be used to compose a chain of thought prompt or a current context prompt

SCENE_OBJECT_PROMPT = "Available scene objects: "
SCENE_OBJECT_RELATIONSHIP_PROMPT = "Object relationships: "
SCENE_PREDICATE_PROMPT = "Available predicates: "
SCENE_PRIMITIVE_PROMPT = "Available primitives: "
HUMAN_INSTRUCTION_PROMPT = "Human instruction: "
EXPLANATION_PROMPT = "Explanation: "
GOAL_PROMPT = "Goal predicate set: "  # set seems to work better than list? nevermind ... hmm
ROBOT_PROMPT = "Robot action sequence: "

def register_openai_key(key: str) -> None:
    openai.api_key = key



def generate_current_setting_prompt(
    instruction: str,
    objects: List[str],
    predicates: List[str],
    primitives: List[str],
    scene_objects: str,
    scene_object_relationships: str
) -> CurrentExample:
    """
    Generate a current example prompt.

    Args:
        instruction (str): natural language instruction
        objects (List[str]): the objects in the current scene
        predicates (List[str]): the predicates in the current scene
        primitives (List[str]): the primitives in the current scene
        scene_objects (str): the objects in the scene
        scene_object_relationships (str): the object relationships in the scene

    Returns:
        CurrentExample: the current example prompt
    """
    return CurrentExample(
        instruction=instruction,
        objects=objects,
        predicates=predicates,
        primitives=primitives,
        scene_objects=scene_objects,
        scene_object_relationships=scene_object_relationships,
    )

def gpt3_call(
    engine: str, 
    overall_prompt: str, 
    max_tokens: int, 
    temperature: float,
    logprobs: int, 
    echo: bool,
    stop: Optional[Union[List[str], str]] = None,
    lm_cache: Optional[Dict] = None,
) -> Tuple[Result, Optional[Dict[Tuple, Any]]]:
    id = tuple((engine, overall_prompt, max_tokens, temperature, logprobs, echo, str(stop)))
    if lm_cache is not None and id in lm_cache.keys():
        print('cache hit, returning')
        response = lm_cache[id]
    else:            
        response = openai.Completion.create(
            engine=engine,
            prompt=overall_prompt,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
            max_tokens=max_tokens,
            stop=stop,
        )
        if lm_cache is not None:
            lm_cache[id] = response

    return response, lm_cache


def generate_lm_response(
    header_prompt: InContextExample,
    current_prompt: CurrentExample,
    engine: str,
    examples: Optional[List[InContextExample]] = None,
    max_tokens: int = 100,
    temperature: float = 0,
    logprobs: int = 0,
    echo: bool = False,
    lm_cache: Optional[Dict[Tuple, Any]] = None,
) -> Tuple[Result, Dict[Tuple, Any]]:
    result = Result(
        header_prompt=header_prompt,
        examples=examples,
        scene_objects=current_prompt.scene_objects
        if current_prompt.use_scene_objects
        else None,
        scene_object_relationships=current_prompt.scene_object_relationships
        if current_prompt.use_scene_object_relationships
        else None,
        predicates=current_prompt.predicates if current_prompt.use_predicates else None,
        primitives=current_prompt.primitives if current_prompt.use_primitives else None,
        human=current_prompt.human if current_prompt.use_human else None,
        engine=engine,
        test_goal=current_prompt.use_goal,
        test_robot=current_prompt.use_robot,
    )

    overall_prompt = ""
    if header_prompt:
        if header_prompt.use_predicates:
            overall_prompt += f"{SCENE_PREDICATE_PROMPT}{header_prompt.predicates}\n"
        if header_prompt.use_primitives:
            overall_prompt += f"{SCENE_PRIMITIVE_PROMPT}{header_prompt.primitives}\n"

    if examples is not None:
        for example in examples:
            overall_prompt += example.overall_example
    import ipdb;ipdb.set_trace()
    if current_prompt.use_predicates:
        overall_prompt += f"{SCENE_PREDICATE_PROMPT}{current_prompt.predicates}\n"
    if current_prompt.use_primitives:
        overall_prompt += f"{SCENE_PRIMITIVE_PROMPT}{current_prompt.primitives}\n"
    if current_prompt.use_scene_objects:
        overall_prompt += f"{SCENE_OBJECT_PROMPT}{current_prompt.scene_objects}\n"
    if current_prompt.use_scene_object_relationships:
        overall_prompt += f"{SCENE_OBJECT_RELATIONSHIP_PROMPT}{current_prompt.scene_object_relationships}\n"
    if current_prompt.use_human:
        overall_prompt += f"{HUMAN_INSTRUCTION_PROMPT}{current_prompt.human}\n"
    if current_prompt.use_goal:
        overall_prompt += f"{GOAL_PROMPT}{current_prompt.goal}\n"

    if current_prompt.predict_explanation:
        overall_prompt += EXPLANATION_PROMPT
        stop = [GOAL_PROMPT, ROBOT_PROMPT]
        response, lm_cache = gpt3_call(
            engine=engine,
            overall_prompt=overall_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            lm_cache=lm_cache,
        )

        # how brittle is this process of parsing the response? what if the LLM decides not to use the word "goal predicate list: "?
        explanation = response["choices"][0]["text"]
        current_prompt.explanation_predicted = explanation
        result.explanation_predicted = explanation
        print("overall prompt: ", overall_prompt)
        print("response: ", response)

        overall_prompt += explanation

    if current_prompt.predict_goal:
        overall_prompt += GOAL_PROMPT
        stop = [ROBOT_PROMPT, SCENE_OBJECT_PROMPT, SCENE_PRIMITIVE_PROMPT]
        response, lm_cache = gpt3_call(
            engine=engine,
            overall_prompt=overall_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            lm_cache=lm_cache,
        )
        goal = response["choices"][0]["text"]
        current_prompt.goal_predicted = goal
        result.goal_predicted = goal
        result.goal_ground_truth = current_prompt.goal

        overall_prompt += goal
        success = check_goal_predicates_equivalent(
            result.goal_ground_truth, result.goal_predicted
        )
        result.goal_success = success

    if current_prompt.predict_robot:
        overall_prompt += ROBOT_PROMPT
        stop = [SCENE_OBJECT_PROMPT, SCENE_PRIMITIVE_PROMPT]
        response, lm_cache = gpt3_call(
            engine=engine,
            overall_prompt=overall_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            lm_cache=lm_cache,
        )
        robot = response["choices"][0]["text"]
        current_prompt.robot_predicted = robot
        result.robot_predicted = robot
        result.robot_ground_truth = current_prompt.robot

        overall_prompt += robot
        success = check_task_plan_result(
            current_prompt.goal, str(result.robot_predicted), current_prompt.pddl_domain_file, current_prompt.pddl_problem_file
        )   
        result.goal_success = success

    print("overall prompt: ", overall_prompt)
    return result, lm_cache
    
    
def get_examples_from_json_dir(path: str) -> List[InContextExample]:
    """
    Load a list of InContextExamples from a directory of json files.

    Note: we assume that the json files are lists of InContextExamples.    
    """
    examples = []
    # using pathlib, get all json files in the directory and load them
    for file in pathlib.Path(path).iterdir():
        if file.suffix == '.json':
            with open(file, 'r') as f:
                examples.extend([InContextExample(**ex) for ex in json.load(f)])

    return examples

def get_examples_from_json(json_file: str) -> List[InContextExample]:
    """Load all examples from the json file.
    
    Args:
        data (dict): The json file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    examples = []
    defaults = data["defaults"]
    for episode in data["examples"]:
        predicates = episode["predicates"] if episode.get("predicates", None) else defaults["predicates"]
        primitives = episode["primitives"] if episode.get("primitives", None) else defaults["primitives"]
        scene = episode["scene"] if episode.get("scene", None) else defaults["scene"]
        human = episode["human"]
        explanation = episode["explanation"] if episode.get("explanation", None) else None
        goal = episode["goal"] if episode.get("goal", None) else None
        robot = episode["robot"] if episode.get("robot", None) else None
        examples.append(InContextExample(predicates, primitives, scene, human, explanation, goal, robot))
    return examples

def predicate_scheme_to_python_syntax(scheme_syntax: str) -> str:
    """
    pddl.goal is a string in scheme-like syntax.
    This function converts it to a python-like syntax.
    
    Examples:
        (and (on ?x ?y) (on ?y ?z)) -> (on(x, y) and on(y, z))
        (on box table) -> "on(box, table)"

    N.B. can use pysumbolic's parse_propositional() to parse the 'python-syntax' string
    """
    prop = scheme_syntax.replace(" ", ", ")
    prop = prop.replace(", ", "(", 1)
    prop = prop + ")"
    return prop

# This is more pysymbolic utils i.e. PDDL utils?
def check_goal_predicates_equivalent(expected_predicates: List[str], predicted_predicates: str) -> bool:
    """
    Check if the predicates in the goal are equivalent.
    
    Note: this method assumes that the predicates are have the
    string representation of a list of strings representing predicates. 
    """
    expected_set = set(expected_predicates)
    predicted_set = set(ast.literal_eval(predicted_predicates.strip()))

    if expected_set == predicted_set:
        return True
    elif predicted_set.issubset(expected_set):
        return True
    else:
        return False

# This is more pysymbolic utils i.e. PDDL utils?
def check_task_plan_result(expected_predicates: List[str], task_plan: str, pddl_domain_file: str, pddl_problem_file: str) -> bool:
    """
    Check if the predicates in the goal are equivalent.
    
    Note: this method assumes that the predicates are have the
    string representation of a list of strings representing predicates. 
    """
    expected_set = set(expected_predicates)
    # strip the task plan of the 'task_plan' prefix
    task_plan = ast.literal_eval(task_plan.strip())
    pddl = symbolic.Pddl(pddl_domain_file, pddl_problem_file)

    # apply the task plan to the problem
    state = pddl.initial_state
    for action in task_plan:
        if action not in pddl.list_valid_actions(state):
            print(f"Action {action} is not valid in state {state}")
            return False
        else:
            state = pddl.next_state(state, action)
    predicted_set = state

    if expected_set == predicted_set:
        print(f"Action sequence {task_plan} is valid and results in the expected state {expected_set}")
        return True
    elif predicted_set.issubset(expected_set):
        print(f"Action sequence {task_plan} is valid and results in a subset of the expected state {expected_set}")
        return True
    else:
        print(f"Action sequence {task_plan} is valid but results in a different state {predicted_set}")
        return False