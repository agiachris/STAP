from dataclasses import dataclass
import dataclasses
from functools import cached_property
from typing import Literal

from temporal_policies.task_planners.lm_utils import APIType


@dataclass
class PDDLConfig:
    pddl_root_dir: str = "configs/pybullet/envs/official/domains"
    pddl_domain: Literal["constrained_packing", "hook_reach"] = "constrained_packing"
    pddl_problem_prefix: str = "new_tamp_problem"
    pddl_file_name: str = "tamp0_domain.pddl"
    _pddl_problem_file: str = ""

    @cached_property
    def pddl_domain_file(self) -> str:
        return self.pddl_root_dir + "/" + self.pddl_domain + "/" + self.pddl_file_name

    @property
    def pddl_problem_file(self) -> str:
        assert self._pddl_problem_file != ""
        return self._pddl_problem_file
    
    @pddl_problem_file.setter
    def pddl_problem_file(self, value: str) -> None:
        self._pddl_problem_file = value

    def get_problem_file(self, problem_name: str) -> str:
        problem_file = (
            self.pddl_root_dir + "/" + self.pddl_domain + "/" + problem_name + ".pddl"
        )
        self.pddl_problem_file = problem_file
        return problem_file

    def get_instructions_file(self, problem_name: str) -> str:
        return (
            self.pddl_root_dir
            + "/"
            + self.pddl_domain
            + "/"
            + problem_name
            + "_instructions.txt"
        )

    def get_prompt_file(self, problem_name: str) -> str:
        return (
            self.pddl_root_dir
            + "/"
            + self.pddl_domain
            + "/"
            + problem_name
            + "_prompt.json"
        )


@dataclass
class ProblemGenerationConfig:
    pddl_cfg: PDDLConfig = dataclasses.field(default_factory=PDDLConfig)
    num_problems: int = 20
    num_workers: int = 1
    min_steps: int = 10
    max_steps: int = 20
    overwrite: bool = False


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
    custom_robot_answer_format: Literal["python_list", "python_list_of_lists"] = "python_list"

@dataclass
class CurrentExampleConfig(InContextExampleConfig):
    predict_goal: bool = False
    predict_explanation: bool = False
    predict_robot: bool = False


@dataclass
class LMConfig:
    engine: Literal["davinci", "curie", "babbage", "ada"] = "curie"
    temperature: float = 0
    logprobs: int = 1
    echo: bool = False
    api_type: APIType = APIType.HELM
    max_tokens: int = 100

@dataclass
class PromptConfig:
    # tricky to define the structure of the prompt
    header_cfg: InContextExampleConfig = InContextExampleConfig()
    single_in_context_prompt_cfg: InContextExampleConfig = (
        InContextExampleConfig()
    )
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
            assert self.prompt_cfg.current_prompt_cfg.use_goal == False, "Cannot predict goal if use_goal is True"
        if self.prompt_cfg.current_prompt_cfg.predict_robot:
            assert self.prompt_cfg.current_prompt_cfg.use_robot == False, "Cannot predict robot if use_robot is True"