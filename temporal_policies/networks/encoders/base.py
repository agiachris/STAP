import abc

import gym  # type: ignore
import torch  # type: ignore

from temporal_policies import envs


class Encoder(torch.nn.Module, abc.ABC):
    """Base critic class."""

    def __init__(self, env: envs.Env, state_space: gym.spaces.Space):
        super().__init__()

        self._env = env
        self._state_space = state_space

    @property
    def env(self) -> envs.Env:
        return self._env

    @property
    def state_space(self) -> gym.spaces.Space:
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
