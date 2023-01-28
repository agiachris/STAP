import pathlib
from typing import Dict, List, Literal, Optional, Union
import numpy as np

from helm.common.authentication import Authentication
from helm.common.request import Token

from configs.base_config import LMConfig
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.task_planners.lm_data_structures import (
    SCENE_OBJECT_PROMPT,
    APIType,
    CurrentExample,
    InContextExample,
)
from temporal_policies.task_planners.lm_utils import (
    authenticate,
    generate_lm_response,
    save_lm_cache,
)
from temporal_policies.envs.pybullet.table import predicates


def get_task_plans_from_lm(
    instruction: str,
    goal: List[str],  # predicted or "ground truth" goal?
    objects: List[str],
    object_relationships: List[str],
    object_relationships_history: List[List[str]],
    executed_actions: List[str],
    pddl_domain_file: str,
    pddl_problem_file: str,
    examples: Optional[List[InContextExample]] = None,
    custom_in_context_example_robot_prompt: str = "Top 1 robot action sequences: ",
    custom_in_context_example_robot_format: Literal[
        "python_list_of_lists", "python_list", "saycan_done", "python_list_with_stop"
    ] = "python_list",
    custom_robot_prompt: str = "Top 4 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: str = "python_list_of_lists",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> List[Union[List[str], List[List[str]], Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)"],
        primitives=["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"]
        + ["stop()"]
        if custom_in_context_example_robot_format == "python_list_with_stop"
        else ["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"],
        use_primitives=True,
        use_predicates=True,
    )
    # assert custom_in_context_example_robot_format == custom_robot_action_sequence_format, (
    #     "custom_in_context_example_robot_format and custom_robot_action_sequence_format "
    #     "must be the same for next action prediction"
    # )
    object_relationships_str = [str(prop) for prop in object_relationships]
    current_prompt = CurrentExample(
        scene_objects=objects,
        scene_object_relationships=object_relationships_str,  # the scene object relationships to be added at the beginning of the prompt
        human=instruction,
        goal_predicted=goal,
        use_scene_objects=True,
        use_scene_object_relationships=True,
        all_prior_object_relationships=object_relationships_history,
        all_executed_actions=executed_actions,
        use_action_object_relationship_history=True,
        use_human=True,
        use_goal=True,
        use_predicted_goal=True,
        predict_robot=True,
        custom_robot_prompt=custom_robot_prompt,
        custom_robot_action_sequence_format=custom_robot_action_sequence_format,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_file=pddl_problem_file,
    )
    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True
        example.use_robot = True
        example.custom_robot_prompt = custom_in_context_example_robot_prompt
        example.custom_robot_action_sequence_format = (
            custom_in_context_example_robot_format
        )
    lm_cfg.engine = "text-davinci-003"
    lm_cfg.api_type = APIType.HELM
    try:
        results, lm_cache = generate_lm_response(
            header_prompt,
            current_prompt,
            examples,
            lm_cfg=lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
            verbose=verbose,
            custom_stop_sequence=[
                "Executed action: ",
                "```",
                custom_in_context_example_robot_prompt,
                "New",
            ],
        )
    except Exception as e:
        print("error", e)
        results, lm_cache = generate_lm_response(
            header_prompt,
            current_prompt,
            examples,
            lm_cfg=lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
            verbose=verbose,
        )
    lm_cfg.engine = "code-davinci-002"
    return (
        results.parsed_robot_predicted,
        lm_cache,
    )


def get_next_actions_from_lm(
    instruction: str,
    goal: List[str],
    objects: List[str],
    object_relationships: List[str],
    all_prior_object_relationships: List[List[str]],
    all_executed_actions: List[str],
    pddl_domain_file: str,
    pddl_problem_file: str,
    examples: Optional[List[InContextExample]] = None,
    custom_in_context_example_robot_prompt: str = "Top robot action sequence: ",
    custom_in_context_example_robot_format: Literal[
        "python_list_with_stop", "python_list"
    ] = "python_list",
    custom_robot_prompt: str = "Top 5 next actions (python list): ",
    custom_robot_action_sequence_format: Literal[
        "python_list_with_stop", "python_list"
    ] = "python_list",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> List[Union[List[str], List[List[str]], Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)", "under(a, b)"],
        primitives=["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"]
        + ["stop()"]
        if custom_in_context_example_robot_format == "python_list_with_stop"
        else ["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"],
        use_primitives=True,
        use_predicates=True,
    )
    assert (
        custom_in_context_example_robot_format == custom_robot_action_sequence_format
    ), (
        "custom_in_context_example_robot_format and custom_robot_action_sequence_format "
        "must be the same for next action prediction"
    )
    object_relationships_str = [str(prop) for prop in object_relationships]
    current_prompt = CurrentExample(
        scene_objects=objects,
        scene_object_relationships=object_relationships_str,
        human=instruction,
        goal_predicted=goal,
        use_scene_objects=True,
        use_scene_object_relationships=True,
        use_human=True,
        use_goal=True,
        use_predicted_goal=True,
        predict_robot=True,
        custom_robot_prompt=custom_robot_prompt,
        custom_robot_action_sequence_format=custom_robot_action_sequence_format,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_file=pddl_problem_file,
        use_action_object_relationship_history=True,
        all_prior_object_relationships=all_prior_object_relationships,
        all_executed_actions=all_executed_actions,
    )

    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True
        example.use_robot = True
        example.custom_robot_prompt = custom_in_context_example_robot_prompt
        example.custom_robot_action_sequence_format = (
            custom_in_context_example_robot_format
        )

    authenticate(APIType.OPENAI, "personal-all")
    lm_cfg.api_type = APIType.OPENAI
    lm_cfg.engine = "code-davinci-002"
    results, lm_cache = generate_lm_response(
        header_prompt,
        current_prompt,
        examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
        custom_stop_sequence=[
            # "Executed action: ",
            "\n",
            custom_robot_prompt,
            "```",
            custom_in_context_example_robot_prompt,
        ],
        verbose=verbose,
    )
    return (
        results.parsed_robot_predicted,
        lm_cache,
    )


def get_action_logprob(
    tokens_predicted: List[Token], n_color_tokens_to_first_action_prompt: int = 1
) -> float:
    """
    Iterate through the tokens in reverse order until we reach
    action_prompt_start (a list of tokens). Then, we can compute the score.

    example:

    action_sequence = ["pick(a)", "place(a, b)", "push(a, hook, rack)", "pull(a, hook)"]
    """
    colon_token_count = 0
    # redo this function except using indices so we can go backwards and check the window size!
    total_logprob = 0
    for i in range(len(tokens_predicted) - 1, -1, -1):
        # print(f'tokens_predicted[{i}].text: {tokens_predicted[i].text} logprob {tokens_predicted[i].logprob}')
        if tokens_predicted[i].text == ":":
            colon_token_count += 1
        if colon_token_count == n_color_tokens_to_first_action_prompt:
            break
        total_logprob += tokens_predicted[i].logprob

    return total_logprob


# TODO(klin): at some point e.g. when we reach the beam search stage, we should
#  probably create a class?
def get_action_scores_from_lm(
    instruction: str,
    potential_actions: List[str],
    goal: List[str],
    objects: List[str],
    object_relationships: List[str],
    all_prior_object_relationships: List[List[str]],
    all_executed_actions: List[str],
    pddl_domain_file: str,
    pddl_problem_file: str,
    score_action_sequence: bool = False,
    examples: Optional[List[InContextExample]] = None,
    custom_in_context_example_robot_prompt: str = "Top robot action sequence: ",
    custom_in_context_example_robot_format: Literal[
        "python_list_with_stop", "python_list"
    ] = "python_list_with_stop",
    custom_robot_prompt: str = "",
    custom_robot_action_sequence_format: Literal[
        "python_list_with_stop", "python_list"
    ] = "python_list_with_stop",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
    lm_cache_file: Optional[str] = None,
    verbose: bool = False,
) -> List[Union[List[float], Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)", "under(a, b)"],
        primitives=["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"]
        + ["stop()"]
        if custom_in_context_example_robot_format == "python_list_with_stop"
        else ["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"],
        use_primitives=True,
        use_predicates=True,
    )
    assert (
        custom_in_context_example_robot_format == custom_robot_action_sequence_format
    ), (
        "custom_in_context_example_robot_format and custom_robot_action_sequence_format "
        "must be the same for next action prediction"
    )
    object_relationships_str = [str(prop) for prop in object_relationships]

    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True
        example.use_robot = True
        example.custom_robot_prompt = custom_in_context_example_robot_prompt
        example.custom_robot_action_sequence_format = (
            custom_in_context_example_robot_format
        )

    original_max_tokens = lm_cfg.max_tokens
    lm_cfg.max_tokens = 0
    lm_cfg.echo = True

    action_logprobs: List[float] = []
    action_logprobs_dct: Dict[str, float] = {}

    logprob_type = f"action sequence" if score_action_sequence else "action"
    current_prompt = CurrentExample(
        scene_objects=objects,
        scene_object_relationships=object_relationships_str,
        human=instruction,
        goal_predicted=goal,
        use_scene_objects=True,
        use_scene_object_relationships=True,
        use_human=True,
        use_goal=True,
        use_predicted_goal=True,
        predict_robot=True,
        custom_robot_prompt=custom_robot_prompt,
        custom_robot_action_sequence_format=custom_robot_action_sequence_format,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_file=pddl_problem_file,
        use_action_object_relationship_history=True,
        all_prior_object_relationships=all_prior_object_relationships,
        all_executed_actions=all_executed_actions,
        score_action=True,
        actions_to_score=potential_actions,
    )
    # generate LM response to the particular prompt (maybe made up of)
    # then, I need to get scores corresponding
    authenticate(APIType.OPENAI, "personal-all")
    lm_cfg.api_type = APIType.OPENAI
    lm_cfg.engine = "code-davinci-002"
    results, lm_cache = generate_lm_response(
        header_prompt,
        current_prompt,
        examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
        verbose=False,
    )
    # TODO(klin): only works for non-helm api
    assert isinstance(
        results.tokens_predicted[0], List
    ), "The LM response is not a list of list of tokens. "
    tokens: Optional[List[List[Token]]] = results.tokens_predicted
    # tokens: Optional[List[Token] = results.tokens_predicted

    if score_action_sequence:
        # assume the action prompting follows the pattern
        # "Executed action: ...\nNew object relationships: ...\nNew action: ..."
        # therefore, we can count the number of ":" tokens to compute the logprobs up to the
        # beginning of the action sequence
        n_colon_tokens_to_action_prompt_start = len(all_executed_actions) * 2 + 1
        action_logprob = get_action_logprob(
            tokens, n_colon_tokens_to_action_prompt_start
        )
    else:
        if isinstance(tokens[0], List):
            for token_list in tokens:
                action_logprob = get_action_logprob(token_list, 1)
                action_logprobs.append(action_logprob)
        else:
            action_logprob = get_action_logprob(tokens, 1)
            action_logprobs.append(action_logprob)

    for i in range(len(potential_actions)):
        potential_action = potential_actions[i]
        action_logprob = action_logprobs[i]
        actions_key = str(
            [str(action).lower() for action in all_executed_actions]
            + [potential_action]
        )
        action_logprobs_dct[actions_key] = action_logprob
        print(f"{logprob_type}: {actions_key}, logprob: {np.round(action_logprob, 3)}")

    save_lm_cache(pathlib.Path(lm_cache_file), lm_cache)

    lm_cfg.echo = False
    lm_cfg.max_tokens = original_max_tokens

    action_logprobs = [logprob for logprob in action_logprobs]

    softmax_beta = 0.3
    softmax_scores = {}
    for option, score in action_logprobs_dct.items():
        softmax_scores[option] = np.exp(score * softmax_beta) / np.sum(
            np.exp(np.array(list(action_logprobs_dct.values())) * softmax_beta)
        )

    print(f"softmax_beta: {softmax_beta}")
    action_scores = [
        softmax_scores[
            str(
                [str(action).lower() for action in all_executed_actions]
                + [potential_action]
            )
        ]
        for potential_action in potential_actions
    ]
    return action_scores, lm_cache


def get_next_action_str_from_lm(
    instruction: str,
    goal: List[str],
    objects: List[str],
    initial_object_relationships: List[str],
    all_prior_object_relationships: List[List[str]],
    all_executed_actions: List[str],
    pddl_domain_file: str,
    pddl_problem_file: str,
    score_action_sequence: bool = False,
    examples: Optional[List[InContextExample]] = None,
    custom_in_context_example_robot_prompt: str = "Top robot action sequence: ",
    custom_in_context_example_robot_format: Literal[
        "python_list_with_stop", "python_list"
    ] = "python_list_with_stop",
    custom_robot_prompt: str = "Executed action (single primitive action string only):",
    custom_robot_action_sequence_format: Literal[
        "python_list_with_stop", "python_list", "str"
    ] = "str",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> List[Union[str, Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)", "under(a, b)"],
        primitives=["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"]
        + ["stop()"]
        if custom_in_context_example_robot_format == "python_list_with_stop"
        else ["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"],
        use_primitives=True,
        use_predicates=True,
    )
    initial_object_relationships_str = [
        str(prop) for prop in initial_object_relationships
    ]

    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True
        example.use_robot = True
        example.custom_robot_prompt = custom_in_context_example_robot_prompt
        example.custom_robot_action_sequence_format = (
            custom_in_context_example_robot_format
        )

    original_max_tokens = lm_cfg.max_tokens
    lm_cfg.max_tokens = 10
    lm_cfg.echo = False

    current_prompt = CurrentExample(
        scene_objects=objects,
        scene_object_relationships=initial_object_relationships_str,
        human=instruction,
        goal_predicted=goal,
        use_scene_objects=True,
        use_scene_object_relationships=True,
        use_human=True,
        use_goal=True,
        use_predicted_goal=True,
        predict_robot=True,
        custom_robot_prompt=custom_robot_prompt,
        custom_robot_action_sequence_format=custom_robot_action_sequence_format,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_file=pddl_problem_file,
        use_action_object_relationship_history=True,
        all_prior_object_relationships=all_prior_object_relationships,
        all_executed_actions=all_executed_actions,
    )
    authenticate(APIType.OPENAI, "personal-all")
    lm_cfg.api_type = APIType.OPENAI
    lm_cfg.engine = "code-davinci-002"
    result, lm_cache = generate_lm_response(
        header_prompt,
        current_prompt,
        examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
        verbose=verbose,
    )

    lm_cfg.echo = False
    lm_cfg.max_tokens = original_max_tokens
    action_str = result.robot_predicted
    return action_str, lm_cache
