import itertools
import pathlib
from typing import Any, Sequence

import imageio  # type: ignore
import numpy as np  # type: ignore

from temporal_policies.envs.base import SequentialEnv
from temporal_policies.envs import utils


class Sequential2D(SequentialEnv):
    """Wrapper around primtive envs for sequential tasks."""

    def __init__(self, env_factories: Sequence[utils.EnvFactory]):
        """Constructs the primtive envs.

        Args:
            env_factories: Ordered list of primitive env factories.
        """
        super().__init__()

        # Construct primitive envs.
        self._envs = []
        for idx_policy, env_factory in enumerate(env_factories):
            if idx_policy == 0:
                env = env_factory()
                env.reset()
            else:
                env = env_factory.cls.load(env, **env_factory.kwargs)
                env_factory.run_post_hooks(env)
            self._envs.append(env)

        self._name = "_".join([env.name for env in self.envs])

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        for env in self.envs:
            env._cumulative_reward = 0.0
            env._steps = 0
            env._physics_steps = 0

        return super().set_state(state)

    def reset(self, idx_policy: int) -> Any:
        """Resets the environment."""
        observation = super().reset(idx_policy)
        for i, env in enumerate(self.envs):
            if i == idx_policy:
                continue
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
