from typing import Sequence

from temporal_policies.planners import shooting
from temporal_policies import agents, dynamics


class RandomShootingPlanner(shooting.ShootingPlanner):
    """A baseline shooting planner that generates actions with a random policy
    and evaluates trajectories using the given policies."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        num_samples: int = 1024,
    ):
        """Constructs a shooting planner with a random policy.

        Args:
            policies: Policies used to evaluate trajectories.
            dynamics: Dynamics model.
            num_samples: Number of shooting samples.
        """
        random_policies = [
            agents.RandomAgent(policy.state_space, policy.action_space)
            for policy in dynamics.policies
        ]

        super().__init__(
            policies=random_policies,
            dynamics=dynamics,
            num_samples=num_samples,
            eval_policies=policies,
        )
