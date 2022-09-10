import abc
import dataclasses
from typing import Optional, Sequence, Union

import numpy as np
import torch

from temporal_policies import agents, dynamics, envs
from temporal_policies.utils import tensors


@dataclasses.dataclass
class PlanningResult:
    actions: np.ndarray  # [T, dim_actions]
    states: np.ndarray  # [T+1, dim_states]
    p_success: float
    values: np.ndarray  # [T]
    visited_actions: Optional[np.ndarray] = None  # [num_visited, T, dim_actions]
    visited_states: Optional[np.ndarray] = None  # [num_visited, T+1, dim_states]
    p_visited_success: Optional[np.ndarray] = None  # [num_visited]
    visited_values: Optional[np.ndarray] = None  # [num_visited, T]


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
        self,
        observation: np.ndarray,
        action_skeleton: Sequence[envs.Primitive],
        return_visited_samples: bool = False,
    ) -> PlanningResult:
        """Plans a sequence of actions following the given action skeleton.

        Args:
            observation: Environment observation.
            action_skeleton: List of (idx_policy, policy_args) 2-tuples.
            return_visited_samples: Whether to return the samples visited during planning.

        Returns:
            Planning result.
        """
        pass
