import ast
import random
from typing import Dict, List, Optional, Union
import numpy as np

from helm.common.authentication import Authentication
from configs.base_config import LMConfig
from temporal_policies.task_planners.lm_data_structures import (
    CurrentExample,
    InContextExample,
)
from temporal_policies.task_planners.lm_utils import (
    generate_lm_response,
    get_examples_from_json_dir,
)
from temporal_policies.envs.pybullet.table import predicates


def get_goal_from_lm(
    instruction: str,
    objects: List[str],
    object_relationships: List[str],
    pddl_domain_file: str,
    pddl_problem_file: str,
    lm_cache: Dict[str, str],
    examples: Optional[List[InContextExample]] = None,
    lm_cfg: Optional[LMConfig] = LMConfig(),
    auth: Optional[Authentication] = None,
) -> List[Union[List[str], Dict[str, str]]]:
    # generate header, in context example, and current example prompts
    header_prompt = InContextExample(
        predicates=["on(a, b)", "inhand(a)", "under(a, b)"],
        primitives=["pick(a)", "place(a, b)", "pull(a, hook)", "push(a, hook, rack)"],
        use_primitives=True,
        use_predicates=True,
    )
    object_relationships_str = [str(prop) for prop in object_relationships]
    current_prompt = CurrentExample(
        scene_objects=objects,
        scene_object_relationships=object_relationships_str,
        human=instruction,
        use_scene_objects=True,
        use_scene_object_relationships=True,
        use_human=True,
        predict_goal=True,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_file=pddl_problem_file,
    )

    for example in examples:
        example.use_scene_objects = True
        example.use_scene_object_relationships = True
        example.use_human = True
        example.use_goal = True

    # generate goal from LM
    results, lm_cache = generate_lm_response(
        header_prompt,
        current_prompt,
        examples,
        lm_cfg=lm_cfg,
        auth=auth,
        lm_cache=lm_cache,
    )

    return results.parsed_goal_predicted, lm_cache


def is_valid_goal_props(
    predicted_props: List[str],
    possible_props: List[predicates.Predicate],
) -> bool:
    possible_props_str = [str(prop) for prop in possible_props]
    syntactically_valid_goals = [prop in possible_props_str for prop in predicted_props]
    if not all(syntactically_valid_goals):
        return False
    else:
        return True
