import abc
from typing import Any, Generic, TypeVar

import gym  # type: ignore

StateType = TypeVar("StateType")


class Env(gym.Env, Generic[StateType]):
    """Base env class with a separate state space for dynamics."""

    @property
    def state_space(self) -> gym.spaces.Space[StateType]:
        """State space."""
        return self.observation_space

    @abc.abstractmethod
    def get_state(self) -> StateType:
        """Gets the environment state."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state: StateType) -> bool:
        """Sets the environment state."""
        raise NotImplementedError

    # TODO: Set idx_policy to first arg.
    @abc.abstractmethod
    def get_observation(self, *args) -> Any:
        """Gets an observation for the current environment state."""
        raise NotImplementedError
