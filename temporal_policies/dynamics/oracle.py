from typing import Any, Optional, Sequence

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import agents, envs
from temporal_policies.dynamics import base as dynamics
from temporal_policies.utils import tensors


class OracleDynamics(dynamics.Dynamics):
    """Dummy dynamics model that uses simulation to get the next state."""

    def __init__(
        self, policies: Sequence[agents.Agent], env: envs.SequentialEnv, device: str = "auto"
    ):
        """Constructs the oracle dynamics wrapper.

        Args:
            policies: Ordered list of all policies.
            env: Oracle environment.
            device: Torch device.
        """
        super().__init__(policies=policies, state_space=env.state_space, device=device)

        if len(self.state_space.shape) != 1:
            raise NotImplementedError(
                "tensors.vmap(self.forward) assumes state_space is 1-dimensional."
            )

        self._env = env

    @property
    def env(self) -> envs.SequentialEnv:
        """Oracle environment."""
        return self._env

    @env.setter
    def env(self, env: envs.SequentialEnv) -> None:
        """Updates the oracle environment."""
        self._env = env

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def forward(
        self,
        state: np.ndarray,
        idx_policy: int,
        action: np.ndarray,
        policy_args: Optional[Any] = None,
    ) -> np.ndarray:
        """Simulates a single step in the environment.

        Args:
            state: Current state.
            idx_policy: Index of executed policy.
            action: Policy action.
            policy_args: Auxiliary policy arguments.

        Returns:
            Next state.
        """
        self.env.set_state(state)
        action = action[: self.policies[idx_policy].action_space.shape[0]]
        self.env.step((action, idx_policy, policy_args))
        next_state: np.ndarray = self.env.get_state()
        return next_state

    @tensors.torch_wrap
    def encode(self, observation: np.ndarray, idx_policy: int) -> np.ndarray:
        """Returns the current environment state.

        WARNING: This ignores the input observation and instead returns the
        environment's current ground truth state. Be careful that the state
        matches the observation as expected.
        """
        env_observation = self.env.get_observation(idx_policy)
        assert observation.ndim == env_observation.ndim
        if (observation != env_observation).any():
            # May happen if self.env is not updated by the dynamics factory.
            raise ValueError("Observation does not match the current env state")
        return self.env.get_state()

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def decode(
        self,
        state: np.ndarray,
        idx_policy: int,
        policy_args: Optional[Any] = None,
    ) -> np.ndarray:
        """Returns the encoded state for the policy.

        Args:
            state: Current state.
            idx_policy: Index of the executed policy.
            policy_args: Auxiliary policy arguments.

        Returs:
            Encoded state for policy.
        """
        self.env.set_state(state)
        observation = self.env.get_observation(idx_policy)
        with torch.no_grad():
            observation = torch.from_numpy(observation).to(self.device)
            policy_state = self.policies[idx_policy].encoder.encode(observation)
            policy_state = policy_state.cpu().numpy()
        return policy_state
