from typing import Any, Optional, Sequence, Union

import torch
import numpy as np

from temporal_policies import agents, networks
from temporal_policies.dynamics import base as dynamics


class RandomDynamics(dynamics.Dynamics):
    """Dynamics model that generates random states."""

    def __init__(self, policies: Sequence[agents.Agent], device: str = "auto"):
        """Constructs the random dynamics.

        Args:
            policies: Ordered list of all policies.
            device: Torch device.
        """
        super().__init__(policies=policies, device=device)

        self._network = networks.Random(
            min=self.state_space.low,
            max=self.state_space.high,
            input_dim=len(self.state_space.shape),
        )

        self.to(self.device)

    @property
    def network(self) -> torch.nn.Module:
        """Random network."""
        return self._network

    def to(self, device: Union[str, torch.device]) -> dynamics.Dynamics:
        """Transfers networks to device."""
        super().to(device)
        try:
            self.network.to(self.device)
        except AttributeError:
            pass
        return self

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Any]],
    ) -> torch.Tensor:
        """Generates a random batched state within the state space.

        Args:
            state: Current state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Next state.
        """
        return self.network(state)
