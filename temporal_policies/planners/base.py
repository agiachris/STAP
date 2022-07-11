import abc
from typing import Any, Sequence, Tuple, Union

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import agents, dynamics
from temporal_policies.utils import tensors


class Planner(abc.ABC):
    """Base planner class."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        device: str = "auto",
    ):
        """Constructs the planner.

        Args:
            policies: Ordered list of policies.
            dynamics: Dynamics model.
            device: Torch device.
        """
        self._policies = policies
        self._dynamics = dynamics
        self.to(device)

    @property
    def policies(self) -> Sequence[agents.Agent]:
        """Ordered list of policies."""
        return self._policies

    @property
    def dynamics(self) -> dynamics.Dynamics:
        """Dynamics model."""
        return self._dynamics

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Planner":
        """Transfers networks to device."""
        self._device = torch.device(tensors.device(device))
        self._dynamics.to(self.device)
        for policy in self.policies:
            policy.to(self.device)
        return self

    @abc.abstractmethod
    def plan(
        self, observation: Any, action_skeleton: Sequence[Tuple[int, Any]]
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Plans a sequence of actions following the given action skeleton.

        Args:
            observation: Environment observation.
            action_skeleton: List of (idx_policy, policy_args) 2-tuples.

        Returns:
            4-tuple (
                actions [T, dim_actions],
                success_probability,
                visited actions [num_visited, T, dim_actions],
                visited success_probability [num_visited])
            ).
        """
        pass
