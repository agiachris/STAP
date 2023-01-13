from typing import Literal, Optional, Dict

import os
import dataclasses
from dataclasses import dataclass, field
from functools import cached_property

from temporal_policies.task_planners.lm_data_structures import APIType


@dataclass
class PDDLConfig:
    domain_dir: str = "configs/pybullet/envs/t2m/official/template"
    domain_file: str = "template_domain.pddl"
    problem_dir: Optional[str] = None
    problem_subdir: Optional[str] = None
    instruction_dir: Optional[str] = None
    prompt_dir: Optional[str] = None
    custom_domain_file: Optional[str] = None
    _pddl_problem_file: Optional[str] = None

    @cached_property
    def pddl_domain_file(self) -> str:
        if self.custom_domain_file is not None:
            return self.custom_domain_file
        return os.path.join(self.domain_dir, self.domain_file)

    @property
    def pddl_problem_file(self) -> str:
        if self._pddl_problem_file is None:
            raise ValueError("Must set PDDLConfig.pddl_problem_file before calling.")
        return self._pddl_problem_file

    @pddl_problem_file.setter
    def pddl_problem_file(self, x: str) -> None:
        self._pddl_problem_file = x

    @property
    def pddl_domain_dir(self) -> str:
        return self.domain_dir

    @property
    def pddl_problem_dir(self) -> str:
        if self.problem_dir is None and self.problem_subdir is None:
            problem_dir = self.domain_dir
        elif self.problem_dir is None and self.problem_subdir is not None:
            problem_dir = os.path.join(self.domain_dir, self.problem_subdir)
        else:
            problem_dir = self.problem_dir
        return problem_dir

    def get_problem_file(self, problem_name: str) -> str:
        problem_dir = self.pddl_problem_dir
        problem_file = os.path.join(problem_dir, problem_name + ".pddl")
        self.pddl_problem_file = problem_file
        return problem_file

    def get_instructions_file(self, problem_name: str) -> str:
        if self.instruction_dir is not None:
            return os.path.join(
                self.instruction_dir, problem_name + "_instructions.txt"
            )
        return os.path.join(self.domain_dir, problem_name + "_instructions.txt")

    def get_prompt_file(self, problem_name: str) -> str:
        if self.prompt_dir is not None:
            return os.path.join(self.prompt_dir, problem_name + "_prompt.json")
        return os.path.join(self.domain_dir, problem_name + "_prompt.json")


@dataclass
class ProblemGenerationConfig:
    pddl_cfg: PDDLConfig = dataclasses.field(default_factory=PDDLConfig)
    num_problems: int = 10
    num_workers: int = 1
    min_steps: int = 10
    max_steps: int = 20
    overwrite: bool = False
    allow_box_on_any_obj: bool = False
    allow_obj_inhand_in_goal: bool = False


@dataclass
class InContextExampleConfig:
    use_predicates: bool = False
    use_primitives: bool = False
    use_scene_objects: bool = (
        False  # if examples have different scenes, then should add the scene ...
    )
    use_scene_object_relationships: bool = False
    use_human: bool = False
    use_explanation: bool = False
    use_goal: bool = False
    use_robot: bool = False
    custom_robot_prompt: str = ""
    custom_robot_action_sequence_format: Literal[
        "python_list", "python_list_of_lists", "saycan_done"
    ] = "python_list"

    predicates: Optional[list] = None
    primitives: Optional[list] = None


@dataclass
class CurrentExampleConfig(InContextExampleConfig):
    predict_goal: bool = False
    predict_explanation: bool = False
    predict_robot: bool = False
    score_robot_action: bool = False


@dataclass
class LMConfig:
    engine: Literal["davinci", "curie", "babbage", "ada"] = "curie"
    temperature: float = 0
    logprobs: int = 1
    echo: bool = False
    api_type: APIType = APIType.HELM
    max_tokens: int = 100

    def __post_init__(self):
        if self.api_type.value == APIType.OPENAI.value:
            engine_dict = {
                # "davinci": "code-davinci-002",
                "davinci": "text-davinci-003",
                "text-davinci-003": "text-davinci-003",
                "text-davinci-002": "text-davinci-002",
                "curie": "text-curie-001",
                "babbage": "text-babbage-001",
                "ada": "text-ada-001",
            }
        elif self.api_type.value == APIType.HELM.value:
            engine_dict = {
                "text-davinci-003": "text-davinci-003",
                "text-davinci-002": "text-davinci-002",
                "davinci": "text-davinci-002",
                "curie": "text-curie-001",
                "babbage": "text-babbage-001",
                "ada": "text-ada-001",
            }
        else:
            raise ValueError("Invalid API type")

        self.engine = (
            engine_dict[self.engine] if self.engine in engine_dict else self.engine
        )


@dataclass
class PromptConfig:
    header_cfg: InContextExampleConfig = field(
        default_factory=lambda: InContextExampleConfig(
            primitives=["pick(a)", "place(a,b)", "push(a, hook)", "pull(a, hook, b)"],
            predicates=["on(a, b)", "under(a, b)"],
        ),
    )
    single_in_context_prompt_cfg: InContextExampleConfig = InContextExampleConfig()
    current_prompt_cfg: CurrentExampleConfig = CurrentExampleConfig()
    lm_cfg: LMConfig = LMConfig()
    n_examples: int = 1


@dataclass
class EvaluationConfig:
    seed: int = 0
    n_evals: int = 2
    prompt_cfg: PromptConfig = PromptConfig()
    pddl_cfg: PDDLConfig = (
        PDDLConfig()
    )  # mainly used to get directory of json files ...
    overwrite_lm_cache: bool = True
    lm_cache_file: str = "lm_cache.pkl"

    def __post_init__(self):
        if self.prompt_cfg.current_prompt_cfg.predict_goal:
            assert (
                self.prompt_cfg.current_prompt_cfg.use_goal == False
            ), "Cannot predict goal if use_goal is True"
        if self.prompt_cfg.current_prompt_cfg.predict_robot:
            assert (
                self.prompt_cfg.current_prompt_cfg.use_robot == False
            ), "Cannot predict robot if use_robot is True"


@dataclass
class PolicyDatasetGenerationConfig:
    """Configuration for generating a dataset of (s, a, s', r) tuples."""

    split: str = "train"
    exp_name: str = "20230113/primitive_dataset"
    custom_path: Optional[str] = None
    # Trainer configs.
    trainer_config: str = "configs/pybullet/trainers/primitive_dataset.yaml"
    agent_config: str = "configs/pybullet/agents/sac.yaml"
    env_config: str = ""
    eval_env_config: str = ""
    encoder_checkpoint: Optional[str] = None
    resume: str = False
    overwrite: str = False
    device: str = "auto"
    seed: int = 0
    gui: int = 0
    use_curriculum: int = 0
    num_pretrain_steps: int = 200000
    num_train_steps: int = 0
    num_eval_episodes: int = 0
    num_env_processes: int = 4
    num_eval_env_processes: int = 0
    # Dataset generation configs.
    pddl_config: PDDLConfig = dataclasses.field(default_factory=lambda: PDDLConfig())
    template_env_yaml: str = (
        "configs/pybullet/envs/t2m/official/template/template_env.yaml"
    )
    primitive: Literal["pick", "place", "push", "pull"] = "pick"
    symbolic_action_type: Literal["valid", "invalid", "all"] = "valid"
    save_env_config: bool = True
    object_types: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "table": "unmovable",
            "rack": "receptacle",
            "hook": "tool",
            "milk": "box",
            "yogurt": "box",
            "icecream": "box",
            "salt": "box",
        }
    )

    @property
    def env_root_dir(self) -> str:
        path = os.path.join(
            "configs/pybullet/envs/t2m/official/primitives", self.exp_name
        )
        return path

    @property
    def env_name(self) -> str:
        return f"{self.split}_{self.symbolic_action_type}_{self.primitive}_{self.seed}"

    @property
    def env_config_path(self) -> str:
        return os.path.join(self.env_root_dir, self.env_name + ".yaml")

    @property
    def path(self) -> str:
        if self.custom_path is not None:
            return self.custom_path
        return os.path.join("models", self.exp_name)

    @property
    def eval_recording_path(self) -> str:
        return os.path.join("plots", self.exp_name)
