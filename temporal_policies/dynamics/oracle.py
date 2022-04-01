from typing import Any, Optional, Sequence

import torch  # type: ignore
import numpy as np  # type: ignore

from temporal_policies import agents, envs
from temporal_policies.dynamics import base as dynamics
from temporal_policies.utils import tensors


class OracleDynamics(dynamics.Dynamics):
    """Dynamics wrapper around an environment."""

    def __init__(self, policies: Sequence[agents.Agent], env: envs.Env):
        """Constructs the oracle dynamics wrapper.

        Args:
            policies: Ordered list of all policies.
            env: Oracle environment.
        """
        super().__init__(policies=policies, state_space=env.state_space)

        if len(self.state_space.shape) != 1:
            raise NotImplementedError(
                "tensors.vmap(self.forward) assumes state_space is 1-dimensional."
            )

        if env.state_space != env.observation_space:
            raise ValueError("Environment state and observation spaces must be the same.")

        self._env = env

    @property
    def env(self) -> envs.Env:
        """Oracle environment."""
        return self._env

    @tensors.torch_wrap
    @tensors.vmap(dim=1)
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
        next_state, _, _, _ = self.env.step((action, idx_policy, policy_args))
        return next_state

    # def forward(
    #     self,
    #     state: torch.Tensor,
    #     idx_policy: torch.Tensor,
    #     action: torch.Tensor,
    #     policy_args: Optional[Any] = None,
    # ) -> torch.Tensor:
    #     state = state.cpu().detach().numpy()
    #     idx_policy = idx_policy.cpu().detach().numpy()
    #     action = action.cpu().detach().numpy()
    #     policy_args = tensors.numpy(policy_args)
    #     if policy_args is not None:
    #         raise NotImplementedError
    #
    #     next_state = np.zeros_like(state)
    #     for t, (s, i, a) in enumerate(zip(state, idx_policy, action)):
    #         self.env.set_state(s)
    #         next_s, _, _, _ = self.env.step((a, i, policy_args))
    #         next_state[t]
    #
    #     next_state = torch.from_numpy(next_state).to(self.device)
    #     return next_state
