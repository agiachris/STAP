from typing import Sequence

from temporal_policies.planners import shooting
from temporal_policies import agents, dynamics


class RandomPlanner(shooting.ShootingPlanner):
    """A non-planner baseline that generates random trajectories."""

    def __init__(self, policies: Sequence[agents.Agent], device: str = "auto"):
        """Initializes random counterparts for the given policies.

        Args:
            policies: Ordered list of (non-random) policies.
            device: Torch device.
        """
        random_policies = [
            agents.RandomAgent(
                action_space=policy.action_space, observation_space=policy.state_space
            )
            for policy in policies
        ]
        random_dynamics = dynamics.RandomDynamics(policies)

        super().__init__(
            policies=random_policies,
            dynamics=random_dynamics,
            num_samples=1,
            device=device,
        )
