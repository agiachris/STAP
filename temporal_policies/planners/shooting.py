import functools
from typing import Sequence, Tuple

import numpy as np
import torch

from temporal_policies import agents, dynamics, envs
from temporal_policies.planners import base as planners
from temporal_policies.planners import utils


class ShootingPlanner(planners.Planner):
    "A shooting planner that generates many trajectories and picks the best one." ""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        num_samples: int = 1024,
        device: str = "auto",
    ):
        """Constructs the shooting planner.

        Args:
            policies: Policies used to generate trajectories.
            dynamics: Dynamics model.
            num_samples: Number of shooting samples.
            device: Torch device.
        """
        super().__init__(policies=policies, dynamics=dynamics, device=device)
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        """Number of shooting samples."""
        return self._num_samples

    def plan(
        self, observation: np.ndarray, action_skeleton: Sequence[envs.Primitive]
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Generates `num_samples` trajectories and picks the best one.

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
        with torch.no_grad():
            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.dynamics.device)
            state = self.dynamics.encode(
                t_observation,
                action_skeleton[0].idx_policy,
                action_skeleton[0].policy_args,
            )
            state = state.repeat(self.num_samples, 1)

            # Roll out trajectories.
            policies = [
                self.policies[primitive.idx_policy] for primitive in action_skeleton
            ]
            states, actions, p_transitions = self.dynamics.rollout(
                state, action_skeleton, policies
            )

            # Evaluate trajectories.
            value_fns = [
                self.policies[primitive.idx_policy].critic
                for primitive in action_skeleton
            ]
            decode_fns = [
                functools.partial(
                    self.dynamics.decode,
                    idx_policy=primitive.idx_policy,
                    policy_args=primitive.policy_args,
                )
                for primitive in action_skeleton
            ]
            p_success = utils.evaluate_trajectory(
                value_fns, decode_fns, states, actions, p_transitions
            )

            # Convert to numpy.
            actions = actions.cpu().numpy()
            p_success = p_success.cpu().numpy()

            # Select best trajectory.
            idx_best = p_success.argmax()
            best_actions = actions[idx_best]
            p_best_success = p_success[idx_best]

        return best_actions, p_best_success, actions, p_success
