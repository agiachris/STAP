import pathlib
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from temporal_policies.envs import base, utils
from temporal_policies.envs.pybox2d.base import Box2DBase
from temporal_policies.utils import random


class Sequential2D(base.Env):
    """Wrapper around primtive envs for sequential tasks."""

    def __init__(self, env_factories: Sequence[utils.EnvFactory]):
        """Constructs the primtive envs.

        Args:
            env_factories: Ordered list of primitive env factories.
        """
        # Construct primitive envs.
        envs: List[Box2DBase] = []
        for idx_policy, env_factory in enumerate(env_factories):
            if idx_policy == 0:
                env = env_factory()
                env.reset()
            else:
                env = env_factory.cls.load(env, **env_factory.kwargs)
                env_factory.run_post_hooks(env)
            env.get_primitive()._idx_policy = idx_policy
            assert isinstance(env, Box2DBase)
            envs.append(env)

        self._envs = envs
        self._envs_dict = {env.name: env for env in self.envs}
        self._current_env = self.envs[0]

        self.name = "_".join([env.name for env in self.envs])
        self.observation_space = self.current_env.observation_space
        self.state_space = self.current_env.state_space
        self.image_space = self.current_env.image_space

    @property
    def envs(self) -> List[Box2DBase]:
        """Primtive envs."""
        return self._envs

    @property
    def current_env(self) -> Box2DBase:
        return self._current_env

    @property
    def action_skeleton(self) -> Sequence[base.Primitive]:
        return [env.get_primitive() for env in self.envs]

    def get_primitive(self) -> base.Primitive:
        return self.current_env.get_primitive()

    def set_primitive(
        self,
        primitive: Optional[base.Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> base.Env:
        if primitive is None:
            primitive = self.get_primitive_info(action_call, idx_policy, policy_args)
        self._current_env = self.envs[primitive.idx_policy]

        return self

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> base.Primitive:
        if action_call is not None:
            return self._envs_dict[action_call].get_primitive_info()
        elif idx_policy is not None:
            return self.envs[idx_policy].get_primitive_info()
        else:
            raise ValueError("Either primitive or action_call must not be None.")

    def get_state(self) -> np.ndarray:
        """Gets the environment state."""
        base_env = self.envs[0]
        return base_env.get_state()

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        for env in self.envs:
            env._cumulative_reward = 0.0
            env._steps = 0
            env._physics_steps = 0

        base_env = self.envs[0]
        return base_env.set_state(state)

    def get_observation(self, idx_policy: Optional[int] = None) -> np.ndarray:
        """Gets an observation for the current state of the environment."""
        if idx_policy is None:
            idx_policy = 0
        return self.envs[idx_policy].get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Executes the step in the current env."""
        return self.current_env.step(action)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Resets the environment."""
        if seed is not None:
            random.seed(seed)

        observation, info = self.current_env.reset()
        for i, env in enumerate(self.envs):
            if env == self.current_env:
                continue
            env.env = self.current_env.env
            env.world = self.current_env.world
            env._t_global = self.current_env._t_global.copy()
            env._setup_spaces()
            env._cumulative_reward = 0.0
            env._steps = 0
            env._physics_steps = 0
            env._render_setup()

        info["seed"] = seed

        return observation, info

    def record_start(
        self,
        prepend_id: Optional[Any] = None,
        frequency: Optional[int] = None,
        mode: str = "default",
    ) -> bool:
        """Starts recording the simulation.

        Args:
            prepend_id: Prepends the new recording with the existing recording
                saved under this id.
            frequency: Recording frequency.
            mode: Recording mode. Options:
                - 'default': record at fixed frequency.
        Returns:
            Whether recording was started.
        """
        return self.current_env.record_start(prepend_id, frequency, mode)

    def record_stop(self, save_id: Optional[Any] = None, mode: str = "default") -> bool:
        """Stops recording the simulation.

        Args:
            save_id: Saves the recording to this id.
            mode: Recording mode. Options:
                - 'default': record at fixed frequency.
        Returns:
            Whether recording was stopped.
        """
        return self.current_env.record_stop(save_id, mode)

    def record_save(
        self,
        path: Union[str, pathlib.Path],
        reset: bool = True,
        mode: Optional[str] = None,
    ) -> bool:
        """Saves all the recordings.

        Args:
            path: Path for the recording.
            reset: Reset the recording after saving.
            mode: Recording mode to save. If None, saves all recording modes.
        Returns:
            Whether any recordings were saved.
        """
        return self.current_env.record_save(path, reset, mode)
