from typing import Sequence

from temporal_policies import agents, dynamics, envs
from temporal_policies.planners import shooting


class RandomShootingPlanner(shooting.ShootingPlanner):
    """A baseline shooting planner that generates actions with a random policy
    and evaluates trajectories using the given policies."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        env: envs.Env,
        num_samples: int = 1024,
        device: str = "auto",
    ):
        """Constructs a shooting planner with a random policy.

        Args:
            policies: Policies used to evaluate trajectories.
            dynamics: Dynamics model.
            env: Policy environment
            num_samples: Number of shooting samples.
            device: Torch device.
        """
        random_policies = [
            agents.RandomAgent(
                env=env_i,
                device=policy.device,
            )
            for policy, env_i in zip(dynamics.policies, env.envs)
        ]

        super().__init__(
            policies=random_policies,
            dynamics=dynamics,
            num_samples=num_samples,
            eval_policies=policies,
            device=device,
        )
