import abc
from typing import Generic, TypeVar

import gym  # type: ignore

StateType = TypeVar("StateType")


class Env(gym.Env, Generic[StateType]):
    """Base env class with a separate state space for dynamics."""

    @property
    def state_space(self) -> gym.spaces.Space[StateType]:
        """State space."""
        return self.observation_space

    @abc.abstractmethod
    def set_state(self, state: StateType) -> bool:
        """Sets the state of the environment."""
        raise NotImplementedError
