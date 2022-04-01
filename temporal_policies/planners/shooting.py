from typing import Any, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from temporal_policies import agents, dynamics
from temporal_policies.planners import base as planners


class ShootingPlanner(planners.Planner):
    "A shooting planner that generates many trajectories and picks the best one." ""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        num_samples: int = 1024,
        eval_policies: Optional[Sequence[agents.Agent]] = None,
    ):
        """Constructs the shooting planner.

        Args:
            policies: Policies used to generate trajectories.
            dynamics: Dynamics model.
            num_samples: Number of shooting samples.
            eval_policies: Optional policies for evaluation. Default uses `policies`.
        """
        super().__init__(policies=policies, dynamics=dynamics)
        self._num_samples = num_samples
        self._eval_policies = policies if eval_policies is None else eval_policies

    @property
    def num_samples(self) -> int:
        """Number of shooting samples."""
        return self._num_samples

    @property
    def eval_policies(self) -> int:
        """Policies for trajectory evaluation."""
        return self._eval_policies

    def plan(
        self, observation: Any, action_skeleton: Sequence[Tuple[int, Any]]
    ) -> Tuple[np.ndarray, float]:
        """Generates `num_samples` trajectories and picks the best one.

        Args:
            observation: Environment observation.
            action_skeleton: List of (idx_policy, policy_args) 2-tuples.

        Returns:
            2-tuple (actions [T, dim_actions], success_probability).
        """
        # Roll out trajectories.
        state = self.dynamics.encode(observation).repeat(self.num_samples, -1)
        policies = [self.policies[idx_policy] for idx_policy, _ in action_skeleton]
        states, actions, p_transitions = self.dynamics.rollout(
            state, action_skeleton, policies
        )

        # Evaluate trajectories.
        value_fns = [
            self.eval_policies[idx_policy].critic for idx_policy, _ in action_skeleton
        ]
        p_success = planners.evaluate_trajectory(
            value_fns, states, actions, p_transitions
        )

        # Select best trajectory.
        idx_best = p_success.argmax()
        best_actions = actions[:, idx_best]
        p_best_success = p_success[idx_best]

        return best_actions, p_best_success
