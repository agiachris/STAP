import ast
import random
from typing import Dict, List, Optional, Union
import numpy as np

from helm.common.authentication import Authentication
from helm.common.request import Token

from configs.base_config import LMConfig
from temporal_policies import envs
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.task_planners.lm_data_structures import CurrentExample, InContextExample, OverallPrompt
from temporal_policies.task_planners.lm_utils import APIType, generate_lm_response, get_examples_from_json_dir
from temporal_policies.envs.pybullet.table import predicates

from temporal_policies.evaluation.utils import get_goal_props_instantiated, get_object_relationships, get_possible_props, get_task_plan_primitives_instantiated


def get_task_plans_from_lm(
    instruction: str,
    goal: List[str],  # predicted or "ground truth" goal?
    objects: List[str],
    object_relationships: List[str],
    pddl_domain_file: str,
    pddl_problem_file: str,
    examples: Optional[List[InContextExample]] = None,
    custom_in_context_example_robot_prompt: str = "Top 1 robot action sequences: ",
    custom_in_context_example_robot_format: str = "python list of lists",
    custom_robot_prompt: str = "Top 2 robot action sequences (python list of lists): ",
    custom_robot_action_sequence_format: str = "python list of lists",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
) -> List[Union[List[str], List[List[str]], Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)"],
        primitives=["pick(a, b)", "place(a, b)", "pull(a, b)", "push(a, b)"],
        use_primitives=True,
        use_predicates=True,
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
    )

    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True
        example.use_robot = True
        example.custom_robot_prompt = custom_in_context_example_robot_prompt
        example.custom_robot_action_sequence_format = custom_in_context_example_robot_format

    overall_prompt = OverallPrompt(
        header_prompt=header_prompt,
        current_prompt=current_prompt,
        examples=examples,
    )
    results, lm_cache = generate_lm_response(
        header_prompt,
        current_prompt,
        examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
    )
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
    custom_in_context_example_robot_format: str = "python list",
    custom_robot_prompt: str = "Top 5 next actions (python list): ",
    custom_robot_action_sequence_format: str = "python list",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
) -> List[Union[List[str], List[List[str]], Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)"],
        primitives=["pick(a, b)", "place(a, b)", "pull(a, b)", "push(a, b)"],
        use_primitives=True,
        use_predicates=True,
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
        example.custom_robot_action_sequence_format = custom_in_context_example_robot_format

    results, lm_cache = generate_lm_response(
        header_prompt,
        current_prompt,
        examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
    )
    return (
        results.parsed_robot_predicted,
        lm_cache,
    )

def get_action_logprob(tokens_predicted: List[Token], n_color_tokens_to_first_action_prompt: int = 1) -> float:
    """
    Iterate through the tokens in reverse order until we reach
    action_prompt_start (a list of tokens). Then, we can compute the score.
    """
    colon_token_count = 0
    # redo this function except using indices so we can go backwards and check the window size!
    total_logprob = 0
    for i in range(len(tokens_predicted) - 1, -1, -1):
        if tokens_predicted[i].text == ":":
            colon_token_count += 1
        if colon_token_count == n_color_tokens_to_first_action_prompt:
            continue
        total_logprob += tokens_predicted[i].logprob

    # for token in reversed(tokens_predicted):
    #     if token.text == action_prompt_start:
    #         break
        # total_logprob += token.logprob
    return total_logprob
    
# TODO(klin): at some point e.g. when we reach the beam search stage, we should
#  probably create a class?
def get_action_scores_from_lm(
    instruction: str,
    potential_actions: str,
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
    custom_in_context_example_robot_format: str = "python list",
    custom_robot_prompt: str = "",
    custom_robot_action_sequence_format: str = "",
    lm_cfg: LMConfig = LMConfig(),
    auth: Optional[Authentication] = None,
    lm_cache: Optional[Dict[str, str]] = None,
) -> List[Union[List[float], Dict[str, str]]]:
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)"],
        primitives=["pick(a, b)", "place(a, b)", "pull(a, b)", "push(a, b)"],
        use_primitives=True,
        use_predicates=True,
    )
    object_relationships_str = [str(prop) for prop in object_relationships]

    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True
        example.use_robot = True
        example.custom_robot_prompt = custom_in_context_example_robot_prompt
        example.custom_robot_action_sequence_format = custom_in_context_example_robot_format

    original_max_tokens = lm_cfg.max_tokens
    lm_cfg.max_tokens = 0
    lm_cfg.echo = True
    
    action_logprobs: List[float] = []
    for potential_action in potential_actions:
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
            action_to_score=potential_action,
        )
        results, lm_cache = generate_lm_response(
            header_prompt,
            current_prompt,
            examples,
            lm_cfg=lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
        )
        tokens: List[Token] = results.tokens_predicted
        if score_action_sequence:
            # assume the action prompting follows the pattern
            # "Executed action: ...\nNew object relationships: ...\nNew action: ..."
            # therefore, we can count the number of ":" tokens to compute the logprobs up to the
            # beginning of the action sequence
            n_colon_tokens_to_action_prompt_start = len(all_executed_actions) * 2 + 1
            action_logprob = get_action_logprob(tokens, n_colon_tokens_to_action_prompt_start)
        else:
            action_logprob = get_action_logprob(tokens, 1)
            action_logprobs.append(action_logprob)

    lm_cfg.echo = False
    lm_cfg.max_tokens = original_max_tokens

    action_logprobs = [logprob for logprob in action_logprobs]
    action_scores = [-1/logprob for logprob in action_logprobs]
    action_scores = [score / sum(action_scores) for score in action_scores]

    # more principled way (?) but results in winner takes all scores
    # action_scores = [np.exp(logprob) for logprob in action_logprobs]
    # action_scores = [score / sum(action_scores) for score in action_scores]
    return action_scores, lm_cache

def get_tokenized_action_prompt_start(overall_prompt, goal_prompt, action_prompt_start):
    """"""
    # use heuristic that action_prompt start can be something between "goal predicate set: " and then the next ":"
    # another way is to use the fact that we know how many actions there are in the action sequence
    # let's just use the goal predicate set hack for now?
    # should then pass in the custom string used in the action sequence prompt then
    return List[str]

def get_action_sequence_scores_from_lm():
    # same as above, except ACTION_PROMPT_START is different?
    # modify the scoring function from above to handle this case too?
    tokens: List[Token] = results.tokens_predicted
    tokenized_action_prompt_start = ":" # first string is GOAL_PREDICATE PROMPT?
    tokenized_action_prompt_start = get_tokenized_action_prompt_start(results.overall_prompt, goal_prompt)
    # find the first time that goal_predicate set is called


    # Goal predicate set: ['on(red_box, rack)', 'inhand(blue_box)', 'on(rack, table)']
    # Executed action: pick(a, b)
    # New object relationships --- hmm wonder if they'd penalize 'unlikely' new object relationships ... rip hmm
    # unclear how to 'normalize' across action sequences ... hmmm
    # conditioning on the new object relationships can boost up the probability of 'good' actions to take hmm
    # ablate, maybe?

    # could start off prompt with some special prompt 'marker', but that could be annoying ... 
    # maybe end with the last tokens of the goal predicate set prompt?
    action_logprob = get_action_logprob(tokens, tokenized_goal_predicate_set_action_prompt)
    # need get_action_logprob that can get tokens up to "goal predicate set: [on(a, b)]\nexecuted_action"
    # maybe need to tokenize the goal predicate set?
    action_logprobs.append(action_logprob)