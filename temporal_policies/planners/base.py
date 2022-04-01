import abc
from typing import Any, Iterable, Sequence, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import agents, dynamics


def evaluate_trajectory(
    value_fns: Iterable[torch.nn.Module],
    states: torch.Tensor,
    actions: torch.Tensor,
    p_transitions: torch.Tensor,
    q_value: bool = True,
) -> np.ndarray:
    """Evaluates probability of success for the given trajectory.

    Args:
        value_fns: List of T value functions.
        states: [T + 1, batch_dims, state_dims] trajectory states.
        actions: [T, batch_dims, state_dims] trajectory actions.
        p_transitions: [T, batch_dims] transition probabilities.
        q_value: Whether to use state-action values (True) or state values (False).

    Returns:
        [batch_dims] Trajectory success probabilities.
    """
    # Compute step success probabilities.
    p_successes = np.zeros_like(p_transitions)
    if q_value:
        for t, value_fn in enumerate(value_fns):
            p_successes[t] = value_fn(states[t], actions[t])
    else:
        for t, value_fn in enumerate(value_fns):
            p_successes[t] = value_fn(states[t])

    # Discard last transition from T-1 to T, since s_T isn't used.
    p_transitions = p_transitions[:-1]

    # Combine probabilities.
    log_p_success = np.log(p_successes).sum(axis=0) + np.log(p_transitions).sum(axis=0)

    return np.exp(log_p_success)


class Planner(abc.ABC):
    """Base planner class."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
    ):
        """Constructs the planner.

        Args:
            policies: Ordered list of policies.
            dynamics: Dynamics model.
        """
        self._policies = policies
        self._dynamics = dynamics

    @property
    def policies(self) -> Sequence[agents.Agent]:
        """Ordered list of policies."""
        return self._policies

    @property
    def dynamics(self) -> dynamics.Dynamics:
        """Dynamics model."""
        return self._dynamics

    @abc.abstractmethod
    def plan(
        self, observation: Any, action_skeleton: Sequence[Tuple[int, Any]]
    ) -> Tuple[np.ndarray, float]:
        """Plans a sequence of actions following the given action skeleton.

        Args:
            observation: Environment observation.
            action_skeleton: List of (idx_policy, policy_args) 2-tuples.

        Returns:
            2-tuple (actions [T, dim_actions], success_probability).
        """
        pass
