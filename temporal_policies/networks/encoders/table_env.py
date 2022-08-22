from typing import Optional

import gym
import numpy as np
import torch

from temporal_policies import envs
from temporal_policies.networks.encoders import Encoder
from temporal_policies.utils import tensors


class TableEnvEncoder(Encoder):
    """Normalizes observation to the range (-0.5, 0.5)."""

    def __init__(self, env: envs.pybullet.TableEnv):
        observation_space = env.observation_space

        state_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(
                (len(env.get_primitive().policy_args) + 1) * observation_space.shape[1],
            ),
            dtype=np.float32,
        )
        super().__init__(env, state_space)

        self.observation_mid = torch.from_numpy(
            (observation_space.low[0] + observation_space.high[0]) / 2
        )
        self.observation_range = torch.from_numpy(
            observation_space.high[0] - observation_space.low[0]
        )

        self._env = env
        # self._idx_args: Optional[List[int]] = None

    def _apply(self, fn):
        """Ensures members get transferred with NormalizeObservation.to(device)."""
        super()._apply(fn)
        self.observation_mid = fn(self.observation_mid)
        self.observation_range = fn(self.observation_range)
        return self

    @tensors.batch(dims=2)
    def forward(
        self,
        observation: torch.Tensor,
        env: Optional[envs.pybullet.TableEnv] = None,
    ) -> torch.Tensor:
        """Normalizes observation to the range (-0.5, 0.5)."""
        if env is None:
            env = self._env
        primitive = env.get_primitive()
        idx_args = env.get_arg_indices(
            idx_policy=primitive.idx_policy, policy_args=primitive.policy_args
        )

        observation = observation[:, idx_args, ...]
        print(idx_args, observation)
        observation = (observation - self.observation_mid) / self.observation_range
        observation = torch.reshape(observation, (-1, self.state_space.shape[0]))
        return observation
