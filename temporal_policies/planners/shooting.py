import functools
from typing import Sequence

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
        self,
        observation: np.ndarray,
        action_skeleton: Sequence[envs.Primitive],
        return_visited_samples: bool = False,
    ) -> planners.PlanningResult:
        """Generates `num_samples` trajectories and picks the best one.

        Args:
            observation: Environment observation.
            action_skeleton: List of primitives.
            return_visited_samples: Whether to return the samples visited during planning.

        Returns:
            Planning result.
        """
        with torch.no_grad():
            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.dynamics.device)

            # Roll out trajectories.
            t_states, t_actions, p_transitions = self.dynamics.rollout(
                t_observation,
                action_skeleton,
                self.policies,
                batch_size=self.num_samples * len(action_skeleton),
            )

            # Evaluate trajectories.
            value_fns = [
                self.policies[primitive.idx_policy].critic
                for primitive in action_skeleton
            ]
            decode_fns = [
                functools.partial(self.dynamics.decode, primitive=primitive)
                for primitive in action_skeleton
            ]
            p_success, t_values = utils.evaluate_trajectory(
                value_fns, decode_fns, t_states, t_actions, p_transitions
            )

        # Select best trajectory.
        idx_best = p_success.argmax()

        # Convert to numpy.
        actions = t_actions[idx_best].cpu().numpy()
        states = t_states[idx_best].cpu().numpy()
        p_success = p_success[idx_best].cpu().numpy()
        values = t_values[idx_best].cpu().numpy()
        if return_visited_samples:
            visited_actions = t_actions.cpu().numpy()
            visited_states = t_states.cpu().numpy()
            visited_p_success = p_success.cpu().numpy()
            visited_values = t_values.cpu().numpy()
        else:
            visited_actions = None
            visited_states = None
            visited_p_success = None
            visited_values = None

        return planners.PlanningResult(
            actions=actions,
            states=states,
            p_success=p_success,
            values=values,
            visited_actions=visited_actions,
            visited_states=visited_states,
            p_visited_success=visited_p_success,
            visited_values=visited_values,
        )
