from typing_extensions import Annotated

import tyro

from configs.base_config import CurrentExampleConfig, EvaluationConfig, InContextExampleConfig, PDDLConfig, PromptConfig


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

TaskPlanEvalConfig = Annotated[
    EvaluationConfig,
    tyro.conf.subcommand(
        name="task_plan",
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
                    use_robot=True
                ),
                current_prompt_cfg=CurrentExampleConfig(
                    use_scene_objects=True,
                    use_scene_object_relationships=True,
                    use_human=True,
                    use_goal=True,
                    predict_robot=True,
                ),
            ),
            pddl_cfg=PDDLConfig(),
        ),
        description="Evaluate LM task planning (use the 'ground truth' goal) on constrained packing",
    ),
]

