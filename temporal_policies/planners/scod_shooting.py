import functools
from typing import Sequence

import numpy as np
import torch

from temporal_policies import agents, dynamics, envs
from temporal_policies.planners import base as planners
from temporal_policies.planners import utils
from temporal_policies.utils import spaces


class SCODShootingPlanner(planners.Planner):
    """A shooting planner that generates many trajectories and picks the best one."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        dynamics: dynamics.Dynamics,
        num_samples: int = 1024,
        num_filter_per_step: int = 512,
        device: str = "auto",
    ):
        """Constructs the shooting planner.

        Args:
            policies: Policies used to generate trajectories.
            dynamics: Dynamics model.
            num_samples: Number of shooting samples.
            num_filter_per_step: Per-step highest uncertainty trajectories to filter.
            device: Torch device.
        """
        super().__init__(policies=policies, dynamics=dynamics, device=device)
        self._num_samples = num_samples
        self._num_filter_per_step = num_filter_per_step

    @property
    def num_samples(self) -> int:
        """Number of shooting samples."""
        return self._num_samples

    @property
    def num_filter_per_step(self) -> int:
        """Per-step highest uncertainty trajectories to filter."""
        return self._num_filter_per_step

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
        # Prepare action spaces.
        T = len(action_skeleton)
        task_space = spaces.null_tensor(self.dynamics.action_space, (T,))
        for t, primitive in enumerate(action_skeleton):
            action_space = self.policies[primitive.idx_policy].action_space
            action_shape = action_space.shape[0]
            task_space[t, :action_shape] = torch.zeros(action_shape)
        task_space = task_space.to(self.device)
        task_dimensionality = int((~torch.isnan(task_space)).sum())

        # Scale number of samples and SCOD filter scale to task size
        num_samples = self.num_samples * task_dimensionality
        num_filter_per_timestep = self.num_filter_per_step * task_dimensionality

        with torch.no_grad():
            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.dynamics.device)

            # Roll out trajectories.
            t_states, t_actions, p_transitions = self.dynamics.rollout(
                t_observation,
                action_skeleton,
                self.policies,
                batch_size=num_samples,
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
            p_success, t_values, t_values_unc = utils.evaluate_trajectory(
                value_fns,
                decode_fns,
                p_transitions,
                t_states,
                actions=t_actions,
                probabilistic_metric="stddev",
            )

        # Filter out trajectories with the highest uncertainty.
        unc_primitive_idx = t_values_unc.argsort(dim=0, descending=True)
        unc_trajectory_idx = unc_primitive_idx[:num_filter_per_timestep]

        # Select best trajectory.
        p_success[unc_trajectory_idx.flatten().unique()] = float("-Inf")
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
