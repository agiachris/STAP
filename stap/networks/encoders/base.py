import abc
from typing import Any, Optional, Union

import gym
import numpy as np
import torch

from stap import envs


class Encoder(torch.nn.Module, abc.ABC):
    """Base encoder class."""

    def __init__(self, env: envs.Env, state_space: gym.spaces.Box):
        """Sets up the encoder spaces.

        Args:
            state_space: Policy latent state space.
        """
        super().__init__()

        self._state_space = state_space

    @property
    def state_space(self) -> gym.spaces.Box:
        """Policy latent state space."""
        return self._state_space

    @abc.abstractmethod
    def forward(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs,
    ) -> torch.Tensor:
        """Encodes the observation to the policy latent state.

        For VAEs, this will return the latent distribution parameters.

        Args:
            observation: Environment observation.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded policy state.
        """
        pass

    def predict(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs,
    ) -> torch.Tensor:
        """Encodes the observation to the policy latent state.

        Args:
            observation: Environment observation.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded policy state.
        """
        return self.forward(observation, policy_args, **kwargs)


class Decoder(torch.nn.Module, abc.ABC):
    """Base decoder class."""

    def __init__(self, env: envs.Env, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decodes the latent state into an observation.

        Args:
            latent: Encoded latent.

        Returns:
            Decoded observation.
        """
        pass
