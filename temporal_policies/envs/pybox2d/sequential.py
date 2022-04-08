import itertools
import pathlib
from typing import Any, Dict, List, Sequence, Union

import gym  # type: ignore
import imageio  # type: ignore
import numpy as np  # type: ignore

from temporal_policies import envs
from temporal_policies.envs.pybox2d import base as pybox2d


class Sequential2D(envs.Env):
    """Wrapper around primtive envs for sequential tasks."""

    def __init__(self, env_configs: Sequence[Union[str, pathlib.Path, Dict[str, Any]]]):
        """Constructs the primtive envs.

        Args:
            env_configs: Ordered list of env configs.
            **kwargs: Box2DBase args.
        """
        super().__init__()

        # Construct primitive envs.
        self._envs = []
        for idx_policy, env_config in enumerate(env_configs):
            env_factory = envs.EnvFactory(env_config)
            if idx_policy == 0:
                env = env_factory()
                env.reset()
            else:
                env = env_factory.cls.load(env, **env_factory.kwargs)
            self._envs.append(env)

        base_env = self._envs[0]
        assert base_env.env is not None
        self._num_bodies = sum(
            len(self._envs[0]._get_shapes(object_name)) for object_name in base_env.env
        )

        # Construct state space.
        ground = base_env._get_shape("playground", "ground")
        x, y = ground["position"]
        workspace = base_env._get_shape_kwargs("playground")
        w, h = workspace["size"]
        low = np.array([x - 0.5 * w, y, -np.pi / 2, -1e3, -1e3, -1e3], dtype=np.float32)
        high = np.array(
            [x + 0.5 * w, y + h, np.pi / 2, 1e3, 1e3, 1e3], dtype=np.float32
        )
        self._state_space = gym.spaces.Box(
            low=np.tile(low, self.num_bodies),
            high=np.tile(high, self.num_bodies),
        )

    @property
    def envs(self) -> List[pybox2d.Box2DBase]:
        """Primtive envs."""
        return self._envs

    @property
    def state_space(self) -> gym.spaces.Box:
        """State space."""
        return self._state_space

    @property
    def num_bodies(self) -> int:
        """Number of environment bodies."""
        return self._num_bodies

    def get_state(self) -> np.ndarray:
        """Gets the environment state.

        [N * 3] array of mutable body properties (position, angle).
        """
        base_env = self.envs[0]
        return base_env.get_state()

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        for env in self.envs:
            env._cumulative_reward = 0.0
            env._steps = 0
            env._physics_steps = 0
            # env._frame_buffer = []

        base_env = self.envs[0]
        return base_env.set_state(state)

    def step(self, action):
        """Executes the step corresponding to the policy index.

        Args:
            action: 3-tuple (action, idx_policy, policy_args).

        Returns:
            4-tuple (observation, reward, done, info).
        """
        # TODO: Debug.
        action, idx_policy, policy_args = action
        return self.envs[idx_policy].step(action)

    def reset(self) -> None:
        """Resets the environment."""
        for env in self.envs:
            env.reset()

    def get_observation(self, *args) -> Any:
        """Gets an observation for the current state of the environment."""
        idx_policy = args[0]
        return self.envs[idx_policy].get_observation()

    def record_start(self) -> None:
        """Starts recording."""
        for env in self.envs:
            env._buffer_frames = True

    def record_stop(self) -> None:
        """Stops recording."""
        for env in self.envs:
            env._buffer_frames = False

    def record_save(self, path: pathlib.Path, stop: bool = False) -> None:
        """Saves the recording to a file.

        Args:
            path: File path.
            stop: Stop recording after saving.
        """
        frames = list(
            itertools.chain.from_iterable([env._frame_buffer for env in self.envs])
        )
        imageio.mimsave(path, frames)
        if stop:
            self.record_stop()
