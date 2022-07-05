import abc

import gym
import torch


class Encoder(torch.nn.Module, abc.ABC):
    """Base encoder class."""

    def __init__(self, state_space: gym.spaces.Space):
        """Sets up the encoder spaces.

        Args:
            state_space: Policy latent state space.
        """
        super().__init__()

        self._state_space = state_space

    @property
    def state_space(self) -> gym.spaces.Space:
        """Policy latent state space."""
        return self._state_space

    @abc.abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encodes the observation to the policy latent state.

        For VAEs, this will return the latent distribution parameters.

        Args:
            observation: Environment observation.

        Returns:
            Encoded policy state.
        """
        pass

    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        """Encodes the observation to the policy latent state.

        Args:
            observation: Environment observation.

        Returns:
            Encoded policy state.
        """
        return self.forward(observation)


class Decoder(torch.nn.Module, abc.ABC):
    """Base decoder class."""

    @abc.abstractmethod
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decodes the latent state into an observation.

        Args:
            latent: Encoded latent.

        Returns:
            Decoded observation.
        """
        pass
