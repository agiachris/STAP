import copy
from pathlib import Path
import random
from typing import Dict, Union
import pickle

import openai
import numpy as np

import tyro
from configs.lm_eval_config import GoalEvalConfig, TaskPlanEvalConfig
from temporal_policies.task_planners.lm_utils import (
    get_examples_from_json_dir,
    generate_lm_response,
    load_lm_cache,
)


def set_header_prompt_attributes(header_prompt, prompt_cfg) -> None:
    for attr, value in vars(prompt_cfg.header_prompt_cfg).items():
        setattr(header_prompt, attr, value)


def set_current_prompt_attributes(current_prompt, prompt_cfg):
    for attr, value in vars(prompt_cfg.current_prompt_cfg).items():
        setattr(current_prompt, attr, value)
    return current_prompt

def set_single_example_prompt_attributes(single_example_prompt, prompt_cfg):
    for attr, value in vars(prompt_cfg.single_example_prompt_cfg).items():
        setattr(single_example_prompt, attr, value)
    return single_example_prompt

def main(config: Union[GoalEvalConfig, TaskPlanEvalConfig]) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)

    openai_api_key = "***REMOVED***"
    openai.api_key = openai_api_key
    engine_dict = {
        "davinci": "text-davinci-002",
        "curie": "text-curie-001",
        "babbage": "text-babbage-001",
        "ada": "text-ada-001",
    }
    ENGINE = engine_dict[config.prompt_cfg.lm_cfg.engine]
    max_tokens = 100

    lm_cache = load_lm_cache(Path(config.lm_cache_file))

    sucesses = 0
    total = 0

    examples = get_examples_from_json_dir(
        config.pddl_cfg.pddl_root_dir + "/" + config.pddl_cfg.pddl_domain
    )

    assert (
            len(examples) + 1 >= config.prompt_cfg.n_examples
        ), f"Not enough examples to generate prompt with {config.prompt_cfg.n_examples} examples, +1 because we don't want to include the current example in the prompt"
        
    # Don't modify examples in place
    for i in range(min(config.n_evals, len(examples) - 1)):
        current_prompt = copy.deepcopy(examples[i])
        set_current_prompt_attributes(current_prompt, config.prompt_cfg)

        header_prompt = copy.deepcopy(filtered_examples[0])
        set_header_prompt_attributes(header_prompt, config.prompt_cfg)

        filtered_examples = [
            copy.deepcopy(example) for idx, example in enumerate(examples) if idx != i
        ]
        single_example_prompts = random.sample(
            filtered_examples, config.prompt_cfg.n_examples
        )

        for single_example_prompt in single_example_prompts:
            set_single_example_prompt_attributes(single_example_prompt, config.prompt_cfg)

        lm_cfg = config.prompt_cfg.lm_cfg
        lm_cfg.engine = ENGINE
        result, lm_cache = generate_lm_response(
            header_prompt,
            current_prompt,
            single_example_prompts,
            lm_cfg=lm_cfg,
            lm_cache=lm_cache,
        )
        result.save_to_json(f"outputs/{ENGINE}_eval_lm_goal_results.json")

        # compare the goal predicted to the expected goal
        if result.goal_success:
            sucesses += 1
        else:
            print(f"\nexpected: {result.goal_ground_truth}")
            print(f"predicted: {result.goal_predicted}\n")

        total += 1

    # save the lm_cache
    with open(config.lm_cache_file, 'wb') as f:
        pickle.dump(lm_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


    print(f"Success rate: {sucesses}/{total} = {sucesses/total}")


if __name__ == "__main__":
    tyro.cli(main)
