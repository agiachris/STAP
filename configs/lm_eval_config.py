from typing import Union
from typing_extensions import Annotated

import tyro

from configs.base_config import CurrentExampleConfig, EvaluationConfig, InContextExampleConfig, LMConfig, PDDLConfig, PromptConfig


GoalEvalConfig = Annotated[
    EvaluationConfig,
    tyro.conf.subcommand(
        name="goal",
        default=EvaluationConfig(
            prompt_cfg=PromptConfig(
                header_cfg=InContextExampleConfig(
                    use_predicates=True, use_primitives=True
                ),
                single_in_context_prompt_cfg=InContextExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                ),
                current_prompt_cfg=CurrentExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    predict_goal=True,
                ),
            ),
            pddl_cfg=PDDLConfig(),
        ),
        description="Evaluate LM goal prediction on constrained packing",
    ),
]

TaskPlan1PlanEvalConfig = Annotated[
    EvaluationConfig,
    tyro.conf.subcommand(
        name="task-plan-1-plan",
        default=EvaluationConfig(
            prompt_cfg=PromptConfig(
                header_cfg=InContextExampleConfig(
                    use_predicates=True, use_primitives=True
                ),
                single_in_context_prompt_cfg=InContextExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    use_robot=True,
                    custom_robot_action_sequence_format="python_list"
                ),
                current_prompt_cfg=CurrentExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    predict_robot=True,
                    custom_robot_prompt="Robot action sequence (python list): ",
                ),
            ),
            pddl_cfg=PDDLConfig(),
        ),
        description="Evaluate LM task planning (use the 'ground truth' goal) on constrained packing",
    ),
]

TaskPlan2PlansEvalConfig = Annotated[
    EvaluationConfig,
    tyro.conf.subcommand(
        name="task-plan-2-plans",
        default=EvaluationConfig(
            prompt_cfg=PromptConfig(
                lm_cfg=LMConfig(max_tokens=200),
                header_cfg=InContextExampleConfig(
                    use_predicates=True, use_primitives=True
                ),
                single_in_context_prompt_cfg=InContextExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    use_robot=True,
                    custom_robot_prompt="Top 1 robot action sequence: ",
                    custom_robot_action_sequence_format="python_list_of_lists"
                ),
                current_prompt_cfg=CurrentExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    predict_robot=True,
                    custom_robot_prompt="Top 2 robot action sequences (python list of lists): ",
                    custom_robot_action_sequence_format="python_list_of_lists"
                ),
            ),
            pddl_cfg=PDDLConfig(),
        ),
        description="Evaluate LM task planning (use the 'ground truth' goal) on constrained packing",
    ),
]

TaskPlan3PlansEvalConfig = Annotated[
    EvaluationConfig,
    tyro.conf.subcommand(
        name="task-plan-3-plans",
        default=EvaluationConfig(
            prompt_cfg=PromptConfig(
                lm_cfg=LMConfig(max_tokens=300),
                header_cfg=InContextExampleConfig(
                    use_predicates=True, use_primitives=True
                ),
                single_in_context_prompt_cfg=InContextExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    use_robot=True,
                    custom_robot_prompt="Top 1 robot action sequence: ",
                    custom_robot_action_sequence_format="python_list_of_lists"
                ),
                current_prompt_cfg=CurrentExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    predict_robot=True,
                    custom_robot_prompt="Top 3 robot action sequences (python list of lists): ",
                    custom_robot_action_sequence_format="python_list_of_lists"
                ),
            ),
            pddl_cfg=PDDLConfig(),
        ),
        description="Evaluate LM task planning (use the 'ground truth' goal) on constrained packing",
    ),
]

UnionEvalConfigs = Union[GoalEvalConfig, TaskPlan1PlanEvalConfig, TaskPlan2PlansEvalConfig, TaskPlan3PlansEvalConfig]