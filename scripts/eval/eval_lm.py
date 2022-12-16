import copy
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

import openai
import numpy as np

import tyro
from temporal_policies.task_planners.lm_utils import (
    CurrentExample,
    InContextExample,
    Result,
    check_goal_predicates_equivalent,
    check_task_plan_result,
    generate_lm_response,
    get_examples_from_json_dir,
    SCENE_OBJECT_PROMPT,
    SCENE_OBJECT_RELATIONSHIP_PROMPT,
    SCENE_PREDICATE_PROMPT,
    SCENE_PRIMITIVE_PROMPT,
    HUMAN_INSTRUCTION_PROMPT,
    EXPLANATION_PROMPT,
    GOAL_PROMPT,
    ROBOT_PROMPT,
    gpt3_call,
)

from configs.lm_eval_config import GoalEvalConfig, TaskPlanEvalConfig



def main(config: Union[GoalEvalConfig, TaskPlanEvalConfig]) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)


    openai.api_key = openai_api_key
    engine_dict = {
        "davinci": "text-davinci-002",
        "curie": "text-curie-001",
        "babbage": "text-babbage-001",
        "ada": "text-ada-001",
    }
    ENGINE = engine_dict[config.prompt_cfg.lm_cfg.engine]
    max_tokens = 100

    lm_cache_file = Path(config.lm_cache_file)       
    # Check if the lm_cache_file exists
    if not lm_cache_file.exists():
        # If it does not exist, create it
        lm_cache_file.touch()
        lm_cache = {}
    else:
        # If it does exist, load it
        with open(lm_cache_file, 'rb') as f:
            lm_cache = pickle.load(f)

    sucesses = 0
    total = 0

    examples = get_examples_from_json_dir(
        config.pddl_cfg.pddl_root_dir + "/" + config.pddl_cfg.pddl_domain
    )
    # Don't modify examples in place
    for i in range(min(config.n_evals, len(examples) - 1)):
        current_prompt = copy.deepcopy(examples[i])
        filtered_examples = [
            example for example in examples if example != current_prompt
        ]

        assert (
            len(filtered_examples) >= config.prompt_cfg.n_examples
        ), f"Not enough examples to generate prompt with {config.prompt_cfg.n_examples} examples"
        filtered_examples = random.sample(
            filtered_examples, config.prompt_cfg.n_examples
        )

        # set all the attributes of the current prompt to be config.prompt_cfg.current_prompt_cfg
        for attr, value in vars(config.prompt_cfg.current_prompt_cfg).items():
            setattr(current_prompt, attr, value)

        # same for the header prompt
        header_prompt = copy.deepcopy(filtered_examples[0])
        for attr, value in vars(config.prompt_cfg.header_cfg).items():
            setattr(header_prompt, attr, value)

        # same for all the single example prompts
        single_example_prompts = []
        for single_example_prompt in filtered_examples:
            for attr, value in vars(
                config.prompt_cfg.single_in_context_prompt_cfg
            ).items():
                setattr(single_example_prompt, attr, value)
            single_example_prompts.append(single_example_prompt)

        result, lm_cache = generate_lm_response(
            header_prompt,
            current_prompt,
            ENGINE,
            single_example_prompts,
            max_tokens=max_tokens,
            temperature=config.prompt_cfg.lm_cfg.temperature,
            logprobs=config.prompt_cfg.lm_cfg.logprobs,
            echo=config.prompt_cfg.lm_cfg.echo,
            lm_cache=lm_cache,
        )
        result.save_to_json(f"outputs/{ENGINE}_eval_lm_goal_results.json")

        # compare the goal predicted to the expected goal
        if current_prompt.predict_goal:
            if result.goal_success:
                sucesses += 1
            else:
                print(f"\nexpected: {result.goal_ground_truth}")
                print(f"predicted: {result.goal_predicted}\n")

        if current_prompt.predict_robot:
            if result.robot_success:
                sucesses += 1
            else:
                print(f"\nexpected: {result.robot_ground_truth}")
                print(f"predicted: {result.robot_predicted}\n")

        total += 1

        # save the lm_cache
        with open(lm_cache_file, 'wb') as f:
            pickle.dump(lm_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


    print(f"Success rate: {sucesses}/{total} = {sucesses/total}")


if __name__ == "__main__":
    tyro.cli(main)
