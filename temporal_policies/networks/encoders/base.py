import abc

import gym  # type: ignore
import torch  # type: ignore


class Encoder(torch.nn.Module, abc.ABC):
    """Base critic class."""

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

        Args:
            observation: Environment observation.

        Returns:
            Encoded policy state.
        """
        pass
