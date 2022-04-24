import abc
from typing import Any, Generic, List, Optional, TypeVar

import gym  # type: ignore

StateType = TypeVar("StateType")


class Env(gym.Env, Generic[StateType]):
    """Base env class with a separate state space for dynamics."""

    @property
    def name(self) -> str:
        return self._name

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
    def get_observation(self) -> Any:
        """Gets an observation for the current environment state."""
        raise NotImplementedError


class SequentialEnv(Env):
    """Wrapper around a sequence of child envs."""

    @property
    def envs(self) -> List[Env]:
        """Primtive envs."""
        return self._envs

    @property
    def state_space(self) -> gym.spaces.Box:
        """State space."""
        base_env = self.envs[0]
        return base_env.get_state()

    def get_state(self) -> StateType:
        """Gets the environment state."""
        base_env = self.envs[0]
        return base_env.get_state()

    def set_state(self, state: StateType) -> bool:
        """Sets the environment state."""
        base_env = self.envs[0]
        return base_env.set_state(state)

    def get_observation(self, idx_policy: Optional[int] = None) -> Any:
        """Gets an observation for the current state of the environment."""
        if idx_policy is None:
            idx_policy = 0
        return self.envs[idx_policy].get_observation()

    def step(self, action):
        """Executes the step corresponding to the policy index.

        Args:
            action: 3-tuple (action, idx_policy, policy_args).

        Returns:
            4-tuple (observation, reward, done, info).
        """
        action, idx_policy, policy_args = action
        return self.envs[idx_policy].step(action)

    def reset(self, idx_policy: int) -> Any:
        return self.envs[idx_policy].reset()
