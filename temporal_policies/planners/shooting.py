import functools
from typing import Sequence

import numpy as np
import torch

from temporal_policies import agents, dynamics, envs
from temporal_policies.planners import base as planners
from temporal_policies.planners import utils
from temporal_policies.utils import spaces, tensors


class ShootingPlanner(planners.Planner):
    """A shooting planner that generates many trajectories and picks the best one."""

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
            # Prepare action spaces.
            T = len(action_skeleton)
            task_space = spaces.null_tensor(self.dynamics.action_space, (T,))
            task_dimensionality = 0
            for t, primitive in enumerate(action_skeleton):
                action_space = self.policies[primitive.idx_policy].action_space
                action_shape = action_space.shape[0]
                task_space[t, :action_shape] = torch.zeros(action_shape)
                task_dimensionality += action_shape
            task_space = task_space.to(self.device)

            # Scale number of samples to task size
            num_samples = self.num_samples * task_dimensionality

            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.dynamics.device)

            # Prepare minibatches.
            element_size = (2 * T) * int(
                np.prod(self.dynamics.state_space.shape)
                + np.prod(self.dynamics.action_space.shape)
            )
            minibatch_size, num_minibatches = tensors.compute_minibatch(
                num_samples, 4 * element_size
            )

            best_actions = spaces.null(self.dynamics.action_space, (num_minibatches, T))
            best_states = spaces.null(self.dynamics.state_space, (num_minibatches, T + 1))
            best_p_success = np.full(num_minibatches, float("nan"))
            best_values = np.full((num_minibatches, T), float("nan"))
            for idx_minibatch in range(num_minibatches):
                # Roll out trajectories.
                t_states, t_actions = self.dynamics.rollout(
                    t_observation,
                    action_skeleton,
                    self.policies,
                    batch_size=minibatch_size,
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
                p_success, t_values, _ = utils.evaluate_trajectory(
                    value_fns,
                    decode_fns,
                    t_states,
                    actions=t_actions,
                )

                # Select best trajectory.
                idx_best = p_success.argmax()

                # Convert to numpy.
                best_actions[idx_minibatch] = t_actions[idx_best].cpu().numpy()
                best_states[idx_minibatch] = t_states[idx_best].cpu().numpy()
                best_p_success[idx_minibatch] = p_success[idx_best].cpu().numpy()
                best_values[idx_minibatch] = t_values[idx_best].cpu().numpy()
                del t_states, t_actions, p_success, t_values

            if return_visited_samples:
                raise NotImplementedError
                # visited_actions = t_actions.cpu().numpy()
                # visited_states = t_states.cpu().numpy()
                # visited_p_success = p_success.cpu().numpy()
                # visited_values = t_values.cpu().numpy()
            else:
                visited_actions = None
                visited_states = None
                visited_p_success = None
                visited_values = None

        idx_best = best_p_success.argmax()
        actions = best_actions[idx_best]
        states = best_states[idx_best]
        p_success = best_p_success[idx_best]
        values = best_values[idx_best]

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
