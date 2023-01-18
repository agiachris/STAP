"""
Example usage:
PYTHONPATH=. python scripts/eval/eval_lm.py config:goal
"""
import copy
from pathlib import Path
import random
import pickle

import numpy as np

import tyro
from temporal_policies.task_planners.lm_data_structures import CurrentExample
from temporal_policies.task_planners.lm_utils import (
    APIType,
    generate_lm_response,
    get_examples_from_json_dir,
    load_lm_cache,
    register_api_key,
)

from helm.common.authentication import Authentication
from configs.lm_eval_config import UnionEvalConfigs


def main(config: UnionEvalConfigs) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)

    if config.prompt_cfg.lm_cfg.api_type.value == APIType.OPENAI.value:
        api_key = "***REMOVED***"
    elif config.prompt_cfg.lm_cfg.api_type.value == APIType.HELM.value:
        api_key = "***REMOVED***"
    else:
        raise ValueError("Invalid API type")

    auth: Authentication = register_api_key(config.prompt_cfg.lm_cfg.api_type, api_key)

    lm_cache = load_lm_cache(Path(config.lm_cache_file))

    sucesses = 0
    total = 0

    # robot_prediction_result_type: Literal["success: partial", "success", "failure: invalid symbolic action", "failure: misses goal"] = ""
    success = 0
    success_partial = 0
    success_superset = 0
    failure_invalid_symbolic_action = 0
    failure_misses_goal = 0

    examples = get_examples_from_json_dir(
        config.pddl_cfg.pddl_root_dir + "/" + config.pddl_cfg.pddl_domain
    )
    # Don't modify examples in place
    for i in range(min(config.n_evals, len(examples) - 1)):
        current_prompt: CurrentExample = CurrentExample()
        current_prompt.create_from_incontext_example(examples[i])
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
            single_example_prompts,
            lm_cfg=config.prompt_cfg.lm_cfg,
            auth=auth,
            lm_cache=lm_cache,
        )
        result.save_to_json(
            f"outputs/{config.prompt_cfg.lm_cfg.engine}_eval_lm_goal_results.json"
        )

        # compare the goal predicted to the expected goal
        if current_prompt.predict_goal:
            if result.goal_success:
                sucesses += 1
            else:
                print(f"\nexpected: {result.goal_ground_truth}")
                print(f"predicted: {result.goal_predicted}\n")

        if current_prompt.predict_robot:
            # start off with success, then check if success: partial, then check if failure_invalid_symbolic_action, then check if failure_misses_goal
            print(
                f"robot_prediction_result_types: {result.robot_prediction_result_types}"
            )
            if any(
                [
                    robot_prediction_result_type == "success"
                    for robot_prediction_result_type in result.robot_prediction_result_types
                ]
            ):
                success += 1
            elif any(
                [
                    robot_prediction_result_type == "success: partial"
                    for robot_prediction_result_type in result.robot_prediction_result_types
                ]
            ):
                success_partial += 1
            elif any(
                [
                    robot_prediction_result_type == "success: superset"
                    for robot_prediction_result_type in result.robot_prediction_result_types
                ]
            ):
                success_superset += 1
            elif any(
                [
                    robot_prediction_result_type == "failure: invalid symbolic action"
                    for robot_prediction_result_type in result.robot_prediction_result_types
                ]
            ):
                failure_invalid_symbolic_action += 1
            elif any(
                [
                    robot_prediction_result_type == "failure: misses goal"
                    for robot_prediction_result_type in result.robot_prediction_result_types
                ]
            ):
                failure_misses_goal += 1
            else:
                raise ValueError("robot_prediction_result_type not recognized")

            if result.robot_success:
                sucesses += 1
            else:
                print(f"\nexpected: {result.robot_ground_truth}")
                print(f"predicted: {result.robot_predicted}\n")

        total += 1

        # save the lm_cache
        with open(config.lm_cache_file, "wb") as f:
            pickle.dump(lm_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Success rate: {sucesses}/{total} = {sucesses/total}")
    print(f"Success: {success}")
    print(f"Success partial: {success_partial}")
    print(f"Success superset: {success_superset}")
    print(f"Failure invalid symbolic action: {failure_invalid_symbolic_action}")
    print(f"Failure misses goal: {failure_misses_goal}")

    # w.r.t goals: failure case is mostly the inhand conundrum --- boosting number of examples doesn't change anything there (7/10) success rate -
    # failures from asking to hold something and it decides to hold and have object on some other object --- 'fixable' by prompt engineering valid predicates


if __name__ == "__main__":
    tyro.cli(main)
