from typing import Any, Dict, List, Literal

from helm.common.authentication import Authentication
from configs.base_config import LMConfig
from temporal_policies.task_planners.lm_data_structures import InContextExample
from temporal_policies.task_planners.task_plans import (
    get_next_action_str_from_lm,
    get_task_plans_from_lm,
)


class LMPlannerAgent:
    def __init__(
        self,
        instruction: str,
        scene_objects: List[str],
        goal_props_predicted: List[str],
        examples: List[InContextExample],
        lm_cfg: LMConfig,
        auth: Authentication,
        lm_cache: Dict[str, Any],
        pddl_domain_file: str,
        pddl_problem_file: str,
    ) -> None:
        self.instruction = instruction
        self.scene_objects = scene_objects
        self.goal_props_predicted = goal_props_predicted
        self.examples = examples
        self.lm_cfg = lm_cfg
        self.auth = auth
        self.lm_cache = lm_cache
        self.pddl_domain_file = pddl_domain_file
        self.pddl_problem_file = pddl_problem_file

    def get_task_plans(
        self,
        object_relationships_history: List[str],
        executed_actions: List[str],
        custom_in_context_example_robot_prompt: str = None,
        custom_in_context_example_robot_format: str = None,
        custom_robot_prompt: str = None,
        custom_robot_action_sequence_format: str = None,
        verbose: bool = False,
    ) -> List[str]:
        self.custom_in_context_example_robot_prompt = (
            custom_in_context_example_robot_prompt
        )
        self.custom_in_context_example_robot_format = (
            custom_in_context_example_robot_format
        )
        self.custom_robot_prompt = custom_robot_prompt
        self.custom_robot_action_sequence_format = custom_robot_action_sequence_format
        task_plans, lm_cache = get_task_plans_from_lm(
            self.instruction,
            self.goal_props_predicted,
            self.scene_objects,
            object_relationships_history[0],
            object_relationships_history,
            executed_actions,
            self.pddl_domain_file,
            self.pddl_problem_file,
            self.examples,
            custom_in_context_example_robot_prompt,
            custom_in_context_example_robot_format,
            custom_robot_prompt,
            custom_robot_action_sequence_format,
            self.lm_cfg,
            self.auth,
            self.lm_cache,
            verbose=verbose,
        )
        self.lm_cache = lm_cache
        return task_plans

    def get_next_action_str(
        self,
        object_relationships_history: List[str],
        executed_actions: List[str],
        custom_in_context_example_robot_prompt: str = "",
        in_context_example_robot_format: Literal[
            "python_list_with_stop"
        ] = "python_list_with_stop",
        robot_prompt: str = "Executed action: ",
        custom_robot_action_sequence_format: str = "str",
        verbose: bool = False,
    ) -> str:
        """
        Prompt LM for next action to execute. If action is stop(), then we are done.
        """
        initial_object_relationships = object_relationships_history[0]
        lm_output_action, lm_cache = get_next_action_str_from_lm(
            self.instruction,
            self.goal_props_predicted,
            self.scene_objects,
            initial_object_relationships,
            object_relationships_history,
            executed_actions,
            self.pddl_domain_file,
            self.pddl_problem_file,
            examples=self.examples,
            lm_cfg=self.lm_cfg,
            auth=self.auth,
            lm_cache=self.lm_cache,
            custom_in_context_example_robot_format=in_context_example_robot_format,
            custom_robot_action_sequence_format=custom_robot_action_sequence_format,
            custom_robot_prompt=robot_prompt,
            verbose=verbose,
        )
        self.lm_cache = lm_cache
        return lm_output_action
