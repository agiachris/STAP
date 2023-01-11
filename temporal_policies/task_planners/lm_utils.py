import ast
from enum import Enum
import pathlib
import pickle
import random
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import openai
import getpass
import json
from helm.common.request import Request, RequestResult
from helm.common.authentication import Authentication
from helm.proxy.services.remote_service import RemoteService
from helm.proxy.accounts import Account
import numpy as np

import symbolic
from configs.base_config import LMConfig
from temporal_policies import envs
from temporal_policies.envs.pybullet.table import predicates

from temporal_policies.task_planners.lm_data_structures import (
    APIType,
    CurrentExample,
    InContextExample,
    Result,
)

# complication: Episode can be used as a single CoT prompt, or as a prompt for current context
# complication: Human might give an impossible goal ...

# Single prompt unit that can be used to compose a chain of thought prompt or a current context prompt

SCENE_OBJECT_PROMPT = "Available scene objects: "
SCENE_OBJECT_RELATIONSHIP_PROMPT = "Object relationships: "
SCENE_PREDICATE_PROMPT = "Available predicates: "
SCENE_PRIMITIVE_PROMPT = "Available primitives: "
HUMAN_INSTRUCTION_PROMPT = "Human instruction: "
EXPLANATION_PROMPT = "Explanation: "
GOAL_PROMPT = "Goal predicate set (single list of predicates, only use available objects): "  # set seems to work better than list? nevermind ... hmm
ROBOT_PROMPT = "Robot action sequence: "

color_box_to_objects = {
    "red_box": "milk",
    "blue_box": "salt",
    "yellow_box": "yoghurt",
    "cyan_box": "icecream",
}

objects_to_color_box = {
    "milk": "red_box",
    "salt": "blue_box",
    "icecream": "cyan_box",
    "yoghurt": "yellow_box",
}


def register_api_key(
    api_type: APIType = APIType.HELM, api_key: str = ""
) -> Optional[Authentication]:
    if api_key == "":
        api_key = getpass.getpass(prompt="Enter a valid API key: ")
    if api_type.value == APIType.OPENAI.value:
        openai.api_key = api_key
    elif api_type.value == APIType.HELM.value:
        service = RemoteService("https://crfm-models.stanford.edu")
        auth = Authentication(api_key=api_key)
        account: Account = service.get_account(auth)
        print(account.usages)
        return auth
    else:
        raise ValueError(f"api_type {api_type} not supported")

def authenticate(api_type: APIType) -> Optional[Authentication]:
    if api_type == APIType.OPENAI:
        raise ValueError("OpenAI API not supported")
    elif api_type == APIType.HELM:
        with open("../credentials.json", "r") as f:
            api_key = json.load(f)["openaiApiKey"]
        return register_api_key(api_type, api_key)
    else:
        raise ValueError("Invalid API type")

def generate_current_setting_prompt(
    instruction: str,
    objects: List[str],
    predicates: List[str],
    primitives: List[str],
    scene_objects: str,
    scene_object_relationships: str,
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
    api_type: APIType = APIType.HELM,  # either via helm api or via openai api
    auth: Optional[Authentication] = None,
) -> Tuple[Result, Optional[Dict[Tuple, Any]]]:
    assert lm_cache is not None, "lm_cache must be provided"
    id = tuple(
        (engine, overall_prompt, max_tokens, temperature, logprobs, echo, str(stop))
    )
    if stop is None:
        stop = []
    if lm_cache is not None and id in lm_cache.keys():
        print("cache hit, returning")
        response = lm_cache[id]
    else:
        if api_type.value == APIType.OPENAI.value:
            response = openai.Completion.create(
                engine=engine,
                prompt=overall_prompt,
                temperature=temperature,
                logprobs=logprobs,
                echo=echo,
                max_tokens=max_tokens,
                stop=stop,
            )
        elif api_type.value == APIType.HELM.value:
            assert auth is not None, "auth must be provided for helm api"
            service = RemoteService("https://crfm-models.stanford.edu")
            request = Request(
                model=f"openai/{engine}",
                prompt=overall_prompt,
                echo_prompt=echo,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop,
                top_k_per_token=logprobs,
            )
            # print request arguments
            try:
                request_result: RequestResult = service.make_request(auth, request)
            except Exception as e:
                print(e)
                print("retrying")
                time.sleep(5)
                request_result: RequestResult = service.make_request(auth, request)
            response: Dict[str, Dict] = {
                "usage": {
                    "total_tokens": len(request_result.completions[0].tokens),
                },
                "choices": [
                    {
                        "text": request_result.completions[0].text,
                        "logprobs": request_result.completions[0].logprob,
                    }
                ],
            }
            response["tokens"] = request_result.completions[0].tokens
        else:
            raise ValueError(f"api_type {api_type} not supported")

        if lm_cache is not None:
            lm_cache[id] = response

    return response, lm_cache


def generate_lm_response(
    header_prompt: Optional[InContextExample] = None,
    current_prompt: Optional[CurrentExample] = None,
    examples: Optional[List[InContextExample]] = None,
    lm_cfg: Optional[LMConfig] = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
) -> Tuple[Result, Dict[Tuple, Any]]:
    if lm_cache is None:
        assert lm_cache is not None, "lm_cache must be provided to save queries"
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
        engine=lm_cfg.engine,
        test_goal=current_prompt.predict_goal,
        test_robot=current_prompt.predict_robot,
    )

    overall_prompt = ""
    if header_prompt:
        overall_prompt = header_prompt.overall_example

    if examples is not None:
        for example in examples:
            overall_prompt += example.overall_example

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
        overall_prompt += (
            f"{GOAL_PROMPT}{current_prompt.goal_predicted}\n"
            if (
                current_prompt.use_predicted_goal
                and current_prompt.goal_predicted != ""
            )
            else f"{GOAL_PROMPT}{current_prompt.goal}\n"
        )

    if current_prompt.predict_explanation:
        overall_prompt += EXPLANATION_PROMPT
        stop = [GOAL_PROMPT, ROBOT_PROMPT]
        response, lm_cache = gpt3_call(
            engine=lm_cfg.engine,
            overall_prompt=overall_prompt,
            max_tokens=lm_cfg.max_tokens,
            temperature=lm_cfg.temperature,
            logprobs=lm_cfg.logprobs,
            echo=lm_cfg.echo,
            stop=stop,
            lm_cache=lm_cache,
            api_type=lm_cfg.api_type,
            auth=auth,
        )

        # how brittle is this process of parsing the response? what if the LLM decides not to use the word "goal predicate list: "?
        explanation = response["choices"][0]["text"]
        current_prompt.explanation_predicted = explanation
        result.explanation_predicted = explanation
        print("overall prompt: ", overall_prompt)
        print("response: ", response)
        overall_prompt += explanation

    result.use_predicted_goal = current_prompt.use_predicted_goal
    result.goal_ground_truth = current_prompt.goal
    if current_prompt.predict_goal:
        overall_prompt += GOAL_PROMPT
        stop = [ROBOT_PROMPT, SCENE_OBJECT_PROMPT, SCENE_PRIMITIVE_PROMPT]
        response, lm_cache = gpt3_call(
            engine=lm_cfg.engine,
            overall_prompt=overall_prompt,
            max_tokens=lm_cfg.max_tokens,
            temperature=lm_cfg.temperature,
            logprobs=lm_cfg.logprobs,
            echo=lm_cfg.echo,
            stop=stop,
            lm_cache=lm_cache,
            api_type=lm_cfg.api_type,
            auth=auth,
        )
        goal = response["choices"][0]["text"]
        current_prompt.goal_predicted = goal
        result.goal_predicted = goal

        overall_prompt += goal
        if result.goal_ground_truth is not None:
            success = check_goal_predicates_equivalent(
                result.goal_ground_truth, result.goal_predicted
            )
            result.goal_success = success

    if current_prompt.predict_robot:
        overall_prompt += get_robot_action_prompt(current_prompt)

        stop = [
            SCENE_OBJECT_PROMPT,
            "\n\n",
            "```",
            "\nRobot action sequence",
        ]
        response, lm_cache = gpt3_call(
            engine=lm_cfg.engine,
            overall_prompt=overall_prompt,
            max_tokens=lm_cfg.max_tokens,
            temperature=lm_cfg.temperature,
            logprobs=lm_cfg.logprobs,
            echo=lm_cfg.echo,
            stop=stop,
            lm_cache=lm_cache,
            api_type=lm_cfg.api_type,
            auth=auth,
        )
        if response["usage"]["total_tokens"] == lm_cfg.max_tokens:
            print("max tokens reached")
            print(f"text is: {response['choices'][0]['text']}")
            import ipdb

            ipdb.set_trace()

        overall_prompt += response["choices"][0]["text"]
        update_result_current_prompt_based_on_response_robot(
            result, current_prompt, response
        )

    print(f"Overall prompt:\n{overall_prompt}\n")
    return result, lm_cache


def get_robot_action_prompt(current_prompt: CurrentExample):
    """
    Get the robot action prompt for the current situation
    based on current_prompt's settings.
    """
    robot_action_prompt = ""
    if current_prompt.use_action_object_relationship_history:
        assert len(current_prompt.all_executed_actions) + 1 == len(
            current_prompt.all_prior_object_relationships
        )
        for i, executed_action in enumerate(current_prompt.all_executed_actions):
            robot_action_prompt += f"Executed action: {str(executed_action).lower()}\n"
            robot_action_prompt += f"New object relationships: {current_prompt.all_prior_object_relationships[i + 1]}\n"

    if current_prompt.score_action:
        robot_action_prompt += f"Executed action: {current_prompt.action_to_score}"
        return robot_action_prompt

    if current_prompt.custom_robot_prompt != "":
        robot_action_prompt += current_prompt.custom_robot_prompt
    else:
        robot_action_prompt += ROBOT_PROMPT
    return robot_action_prompt


def update_result_current_prompt_based_on_response_robot(
    result: Result,
    current_prompt: CurrentExample,
    response: Dict,
):
    robot = response["choices"][0]["text"]

    current_prompt.robot_predicted = robot
    result.robot_predicted = robot
    result.robot_ground_truth = current_prompt.robot_action_sequence

    robot_prediction_result_types, predicted_task_plan_descriptions = [], []
    if current_prompt.custom_robot_action_sequence_format == "python_list_of_lists":
        parsed_robot_predicted_lst = result.parsed_robot_predicted_list_of_lists
        for parsed_robot_predicted in parsed_robot_predicted_lst:
            (
                robot_prediction_result_type,
                predicted_task_plan_description,
            ) = check_task_plan_result(
                current_prompt.goal_predicted
                if current_prompt.use_predicted_goal or current_prompt.goal is None
                else current_prompt.goal,
                str(parsed_robot_predicted),
                current_prompt.pddl_domain_file,
                current_prompt.pddl_problem_file,
            )
            robot_prediction_result_types.append(robot_prediction_result_type)
            predicted_task_plan_descriptions.append(predicted_task_plan_description)
    elif current_prompt.custom_robot_action_sequence_format == "python_list":
        parsed_robot_predicted = result.parsed_robot_predicted
        (
            robot_prediction_result_type,
            predicted_task_plan_description,
        ) = check_task_plan_result(
            current_prompt.goal_predicted
            if current_prompt.use_predicted_goal or current_prompt.goal is None
            else current_prompt.goal,
            str(parsed_robot_predicted),
            current_prompt.pddl_domain_file,
            current_prompt.pddl_problem_file,
        )
        robot_prediction_result_types.append(robot_prediction_result_type)
        predicted_task_plan_descriptions.append(predicted_task_plan_description)

    result.robot_success = any(
        [
            "success" in robot_prediction_result_type
            for robot_prediction_result_type in robot_prediction_result_types
        ]
    )
    result.robot_prediction_result_types = robot_prediction_result_types
    result.predicted_task_plan_descriptions = predicted_task_plan_descriptions
    result.custom_robot_prompt = current_prompt.custom_robot_prompt
    result.custom_robot_action_sequence_format = (
        current_prompt.custom_robot_action_sequence_format
    )
    result.tokens_predicted = (
        response["tokens"] if response.get("tokens", False) else None
    )


def load_lm_cache(lm_cache_file: pathlib.Path) -> Dict:
    # Check if the lm_cache_file exists
    if not lm_cache_file.exists():
        # If it does not exist, create it
        lm_cache_file.touch()
        lm_cache = {}
    else:
        # If it does exist, load it
        with open(lm_cache_file, "rb") as f:
            # check if the file is empty
            if pathlib.Path(lm_cache_file).stat().st_size == 0:
                lm_cache = {}
            else:
                lm_cache = pickle.load(f)
        if lm_cache is None:
            lm_cache = {}
    return lm_cache


def get_examples_from_json_dir(path: str) -> List[InContextExample]:
    """
    Load a list of InContextExamples from a directory of json files.

    Note: we assume that the json files are lists of InContextExamples.
    """
    examples = []
    # using pathlib, get all json files in the directory and load them
    for file in pathlib.Path(path).iterdir():
        if file.suffix == ".json":
            print(f"Loading examples from {file} ...")
            with open(file, "r") as f:
                examples.extend([InContextExample(**ex) for ex in json.load(f)])

    return examples


def get_examples_from_json(json_file: str) -> List[InContextExample]:
    """Load all examples from the json file.

    Args:
        data (dict): The json file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    examples = []
    defaults = data["defaults"]
    for episode in data["examples"]:
        predicates = (
            episode["predicates"]
            if episode.get("predicates", None)
            else defaults["predicates"]
        )
        primitives = (
            episode["primitives"]
            if episode.get("primitives", None)
            else defaults["primitives"]
        )
        scene = episode["scene"] if episode.get("scene", None) else defaults["scene"]
        human = episode["human"]
        explanation = (
            episode["explanation"] if episode.get("explanation", None) else None
        )
        goal = episode["goal"] if episode.get("goal", None) else None
        robot = episode["robot"] if episode.get("robot", None) else None
        examples.append(
            InContextExample(
                predicates, primitives, scene, human, explanation, goal, robot
            )
        )
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
def check_goal_predicates_equivalent(
    expected_predicates: List[str], predicted_predicates: str
) -> bool:
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
def check_task_plan_result(
    expected_predicates: List[str],
    task_plan: str,
    pddl_domain_file: str,
    pddl_problem_file: str,
) -> Literal[
    "success: partial",
    "success",
    "success: superset",
    "failure: invalid symbolic action",
    "failure: misses goal",
]:
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
            predicted_task_plan_description = (
                f"Action {action} is not valid in state {state}"
            )
            robot_prediction_result_type = "failure: invalid symbolic action"
            print(f"{robot_prediction_result_type}\n{predicted_task_plan_description}")
            return robot_prediction_result_type, predicted_task_plan_description
        else:
            state = pddl.next_state(state, action)
    predicted_set = state

    if expected_set == predicted_set:
        predicted_task_plan_description = f"Action sequence {task_plan} is valid and results in the expected state {expected_set}"
        robot_prediction_result_type = "success"
        print(f"{robot_prediction_result_type}\n{predicted_task_plan_description}")
    elif predicted_set.issubset(expected_set):
        predicted_task_plan_description = f"Action sequence {task_plan} is valid and results in a subset {predicted_set} of the expected state {expected_set}"
        robot_prediction_result_type = "success: partial"
        print(f"{robot_prediction_result_type}\n{predicted_task_plan_description}")
    elif expected_set.issubset(predicted_set):
        predicted_task_plan_description = f"Action sequence {task_plan} is valid and results in a superset {predicted_set} of the expected state {expected_set}"
        robot_prediction_result_type = "success: superset"
        print(f"{robot_prediction_result_type}\n{predicted_task_plan_description}")

    else:
        predicted_task_plan_description = f"Action sequence {task_plan} is valid but results in state {predicted_set} instead of the goal state {expected_set}"
        robot_prediction_result_type = "failure: misses goal"
        print(f"{robot_prediction_result_type}\n{predicted_task_plan_description}")
    return robot_prediction_result_type, predicted_task_plan_description


def save_lm_cache(
    lm_cache_file: Union[str, pathlib.Path], lm_cache: Dict[str, str]
) -> None:
    # save the lm_cache
    with open(lm_cache_file, "wb") as f:
        print(f"Saving lm_cache to {lm_cache_file}...")
        pickle.dump(lm_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    overall_prompt = """Available predicates: ['on(a, b)', 'inhand(a)']

Available primitives: ['pick(a, b)', 'place(a, b)', 'pull(a, b)', 'push(a, b)']

Available scene objects: ['table', 'red_box', 'rack', 'blue_box']
Object relationships: ['on(blue_box, table)', 'on(red_box, blue_box)', 'on(rack, table)']
Human instruction: please place all the boxes on top of the rack and grab the blue_box, thanks!
Goal predicate set: ['on(red_box, rack)', 'inhand(blue_box)', 'on(rack, table)']
Top 1 robot action sequence: [['pick(red_box, blue_box)', 'place(red_box, rack)', 'pick(blue_box, table)'], ]

Available scene objects: ['table', 'cyan_box', 'rack', 'blue_box']
Object relationships: ['on(blue_box, cyan_box)', 'on(rack, table)', 'on(cyan_box, rack)']
Human instruction: please situate the blue_box over the table and stack the cyan_box above the rack please?
Goal predicate set: ['on(blue_box, table)', 'on(rack, table)', 'on(cyan_box, rack)']
Top 1 robot action sequence: [['pick(blue_box, cyan_box)', 'place(blue_box, table)'], ]

Available scene objects: ['table', 'red_box', 'rack', 'blue_box', 'yellow_box', 'cyan_box']
Object relationships: ['on(red_box, rack)', 'on(blue_box, cyan_box)', 'on(yellow_box, blue_box)', 'on(cyan_box, table)', 'on(rack, table)']
Human instruction: could you set the blue_box on the top of the rack and get the cyan_box on top of the table and perch the yellow_box onto the rack and pick up the red_box, thanks!
Goal predicate set: ['on(blue_box, rack)', 'on(cyan_box, table)', 'on(yellow_box, rack)', 'on(rack, table)', 'inhand(red_box)']
Top 1 robot action sequence: [['pick(blue_box, cyan_box)', 'place(blue_box, rack)', 'pick(yellow_box, blue_box)', 'place(yellow_box, rack)', 'pick(red_box, rack)'], ]

Available scene objects: ['table', 'red_box', 'rack', 'yellow_box']
Object relationships: ['on(red_box, rack)', 'on(rack, table)', 'on(yellow_box, table)']
Human instruction: could you set all the boxes over the table - thanks
Goal predicate set: ['on(yellow_box, table)', 'on(rack, table)', 'on(red_box, table)']
Top 1 robot action sequence: [['pick(red_box, rack)', 'place(red_box, table)'], ]

Available scene objects: ['table', 'cyan_box', 'red_box', 'rack', 'yellow_box']
Object relationships: ['on(yellow_box, rack)', 'on(red_box, cyan_box)', 'on(rack, table)', 'on(cyan_box, rack)']
Human instruction: could you position the yellow_box over the table and move the red_box on the rack and hang on to the cyan_box
Goal predicate set: ['on(yellow_box, table)', 'on(red_box, rack)', 'inhand(cyan_box)', 'on(rack, table)']
Top 1 robot action sequence: [['pick(red_box, cyan_box)', 'place(red_box, rack)', 'pick(yellow_box, rack)', 'place(yellow_box, table)', 'pick(cyan_box, rack)'], ]

Available scene objects: ['table', 'red_box', 'rack', 'yellow_box', 'blue_box', 'cyan_box']
Object relationships: ['on(cyan_box, yellow_box)', 'on(red_box, yellow_box)', 'on(blue_box, yellow_box)', 'on(yellow_box, rack)', 'on(rack, table)']
Human instruction: arrange the red_box on top of the rack and stack the blue_box over the table and arrange the cyan_box above the table and put the yellow_box on the top of the rack now, thanks!
Goal predicate set: ['on(red_box, rack)', 'on(blue_box, table)', 'on(cyan_box, table)', 'on(yellow_box, rack)', 'on(rack, table)']
Top 3 robot action sequences (python list of lists):"""
    api_key = "***REMOVED***"
    # api_key = "***REMOVED***"
    auth = register_api_key(api_type=APIType.HELM, api_key=api_key)
    response = gpt3_call(
        "text-davinci-003",
        overall_prompt,
        200,
        0,
        1,
        True,
        api_type=APIType.HELM,
        auth=auth,
    )
    print(response)
    print(response[0]["choices"][0]["text"])
    # api_key = "***REMOVED***"
    # api_key = "***REMOVED***"
    # auth = register_api_key(api_type=APIType.OPENAI, api_key=api_key)
    # response = gpt3_call("text-davinci-003", overall_prompt, 200, 0, 1, True, api_type=APIType.OPENAI, auth=auth, stop=[SCENE_OBJECT_PROMPT])
    # print(response)
    # print(response[0]["choices"][0]["text"])
