from typing import Any, Optional, Union

import gym
import numpy as np
import torch

from temporal_policies import envs
from temporal_policies.networks.encoders import Encoder


class NormalizeObservation(Encoder):
    """Normalizes observation to the range (-0.5, 0.5)."""

    def __init__(self, env: envs.Env):
        observation_space = env.observation_space
        if not isinstance(observation_space, gym.spaces.Box):
            raise NotImplementedError

        state_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=observation_space.shape, dtype=np.float32
        )
        super().__init__(env, state_space)

        self.observation_mid = torch.from_numpy(
            (observation_space.low + observation_space.high) / 2
        )
        self.observation_range = torch.from_numpy(
            observation_space.high - observation_space.low
        )

    def _apply(self, fn):
        """Ensures members get transferred with NormalizeObservation.to(device)."""
        super()._apply(fn)
        self.observation_mid = fn(self.observation_mid)
        self.observation_range = fn(self.observation_range)
        return self

    def forward(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs
    ) -> torch.Tensor:
        """Normalizes observation to the range (-0.5, 0.5)."""
        return (observation - self.observation_mid) / self.observation_range
