from typing import Literal, Optional, Dict

import os
import dataclasses
from dataclasses import dataclass
from functools import cached_property


@dataclass
class PDDLConfig:
    domain_dir: str = "configs/pybullet/envs/official/template"
    domain_file: str = "template_valid_domain.pddl"
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
class PolicyDatasetGenerationConfig:
    """Configuration for generating a dataset of (s, a, s', r) tuples."""

    split: str = "train"
    exp_name: str = "datasets"
    custom_path: Optional[str] = None
    # Trainer configs.
    trainer_config: str = "configs/pybullet/trainers/datasets/primitive_valid_dataset.yaml"
    agent_config: str = "configs/pybullet/agents/single_stage/sac.yaml"
    device: str = "auto"
    seed: int = 0
    # Dataset generation configs.
    pddl_handler: Optional[PDDLConfig] = None
    template_env_yaml: str = (
        "configs/pybullet/envs/official/template/template_env.yaml"
    )
    primitive: Literal["pick", "place", "push", "pull"] = "pick"
    symbolic_action_type: Literal["valid", "invalid"] = "valid"
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
    def pddl_config(self) -> PDDLConfig:
        if isinstance(self.pddl_handler, PDDLConfig):
            return self.pddl_handler
        domain_file = f"template_{self.symbolic_action_type}_domain.pddl"
        self.pddl_handler = PDDLConfig(domain_file=domain_file)
        return self.pddl_handler

    @property
    def env_root_dir(self) -> str:
        path = os.path.join(
            "configs/pybullet/envs/official/primitives", self.exp_name
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
