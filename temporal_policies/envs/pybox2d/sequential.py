import itertools
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np

from temporal_policies.envs import base, utils


class SequentialEnv(base.Env):
    """Wrapper around a sequence of child envs."""

    def __init__(self, envs: List[base.Env]):
        self._envs = envs
        self.state_space = self.envs[0].state_space
        self.name = "_".join([env.name for env in self.envs])

    @property
    def envs(self) -> List[base.Env]:
        """Primtive envs."""
        return self._envs

    def get_state(self) -> np.ndarray:
        """Gets the environment state."""
        base_env = self.envs[0]
        return base_env.get_state()

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        base_env = self.envs[0]
        return base_env.set_state(state)

    def get_observation(self, idx_policy: Optional[int] = None) -> np.ndarray:
        """Gets an observation for the current state of the environment."""
        if idx_policy is None:
            idx_policy = 0
        return self.envs[idx_policy].get_observation()

    def step(
        self, action: Tuple[np.ndarray, int]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Executes the step corresponding to the policy index.

        Args:
            action: 2-tuple (action, idx_policy).

        Returns:
            4-tuple (observation, reward, done, info).
        """
        env_action, idx_policy = action
        return self.envs[idx_policy].step(env_action)

    def reset(self, idx_policy: int) -> np.ndarray:
        return self.envs[idx_policy].reset()


class Sequential2D(SequentialEnv):
    """Wrapper around primtive envs for sequential tasks."""

    def __init__(self, env_factories: Sequence[utils.EnvFactory]):
        """Constructs the primtive envs.

        Args:
            env_factories: Ordered list of primitive env factories.
        """
        # Construct primitive envs.
        envs = []
        for idx_policy, env_factory in enumerate(env_factories):
            if idx_policy == 0:
                env = env_factory()
                env.reset()
            else:
                env = env_factory.cls.load(env, **env_factory.kwargs)
                env_factory.run_post_hooks(env)
            envs.append(env)

        super().__init__(envs)

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        for env in self.envs:
            env._cumulative_reward = 0.0
            env._steps = 0
            env._physics_steps = 0

        return super().set_state(state)

    def reset(self, idx_policy: int) -> np.ndarray:
        """Resets the environment."""
        observation = super().reset(idx_policy)
        for i, env in enumerate(self.envs):
            if i == idx_policy:
                continue
            env.env = self.envs[idx_policy].env
            env.world = self.envs[idx_policy].world
            env._t_global = self.envs[idx_policy]._t_global.copy()
            env._setup_spaces()
            env._cumulative_reward = 0.0
            env._steps = 0
            env._physics_steps = 0
            env._render_setup()

        return observation

    def record_start(self) -> None:
        """Starts recording."""
        for env in self.envs:
            env._buffer_frames = True

    def record_pause(self) -> None:
        """Stops recording."""
        for env in self.envs:
            env._buffer_frames = False

    def record_stop(self) -> None:
        """Stops recording."""
        self.record_pause()
        for env in self.envs:
            env._frame_buffer.clear()

    def record_save(self, path: pathlib.Path, stop: bool = False) -> None:
        """Saves the recording to a file.

        Args:
            path: File path.
            stop: Stop recording after saving.
        """
        frames = list(
            itertools.chain.from_iterable([env._frame_buffer[::3] for env in self.envs])
        )
        imageio.mimsave(path, frames)
        if stop:
            self.record_stop()
