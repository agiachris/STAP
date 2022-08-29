import pathlib
from typing import Any, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np

from temporal_policies.envs.base import Env, Primitive


class VariantEnv(Env):
    def __init__(self, variants: Sequence[Env]):
        self._variants = variants
        self._idx_variant = np.random.randint(len(self.variants))

    @property
    def variants(self) -> Sequence[Env]:
        return self._variants

    @property
    def env(self) -> Env:
        return self.variants[self._idx_variant]

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.spaces.Box:  # type: ignore
        return self.env.observation_space

    @property
    def state_space(self) -> gym.spaces.Box:  # type: ignore
        return self.env.state_space

    @property
    def image_space(self) -> gym.spaces.Box:  # type: ignore
        return self.env.image_space

    @property
    def action_space(self) -> gym.spaces.Box:  # type: ignore
        return self.env.action_space

    @property
    def action_scale(self) -> gym.spaces.Box:
        return self.env.action_scale

    @property
    def action_skeleton(self) -> Sequence[Primitive]:
        return self.env.action_skeleton

    @property
    def primitives(self) -> List[str]:
        return self.env.primitives

    def get_primitive(self) -> Primitive:
        return self.env.get_primitive()

    def set_primitive(
        self,
        primitive: Optional[Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> "Env":
        return self.env.set_primitive(primitive, action_call, idx_policy, policy_args)

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> Primitive:
        return self.env.get_primitive_info(action_call, idx_policy, policy_args)

    def create_primitive_env(self, primitive: Primitive) -> "Env":
        return self.env.create_primitive_env(primitive)

    def get_state(self) -> np.ndarray:
        return self.env.get_state()

    def set_state(self, state: np.ndarray) -> bool:
        return self.env.set_state(state)

    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        return self.env.get_observation(image)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        self._idx_variant = np.random.randint(len(self.variants))
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return self.env.step(action)

    def close(self):
        for env in self.variants:
            env.close()

    def render(self) -> np.ndarray:  # type: ignore
        return self.env.render()

    def record_start(
        self,
        prepend_id: Optional[Any] = None,
        frequency: Optional[int] = None,
        mode: str = "default",
    ) -> bool:
        return self.env.record_start(prepend_id, frequency, mode)

    def record_stop(self, save_id: Optional[Any] = None, mode: str = "default") -> bool:
        return self.env.record_stop(save_id, mode)

    def record_save(
        self,
        path: Union[str, pathlib.Path],
        reset: bool = True,
        mode: Optional[str] = None,
    ) -> bool:
        return self.env.record_save(path, reset, mode)
