import numpy as np
from copy import deepcopy

import torch

from .pybox2d_base import Box2DPlannerBase


class ShootingPlanner(Box2DPlannerBase):
    def __init__(self, samples, standard_deviation=None, sample_policy=False, **kwargs):
        """Perform shooting-based model-predictive control with randomly sampled actions OR actions
        sampled from a multivariate Gaussian distribution centered at the policy's prediction. Return
        the first action of the highest scoring trajectory.

        args:
            samples: number of sampled trajectories
            standard_deviation: float standard deviation parameterizing diagonal Gaussian
            sample_policy: whether or not stochastic policy outputs should be sampled or taken at the mean
        """
        super().__init__(**kwargs)
        self._samples = samples
        self._standard_deviation = standard_deviation
        self._policy_kwargs = {"sample": sample_policy}

    @property
    def planner_settings(self):
        settings = {
            "samples": self._samples,
            "standard_deviation": self._standard_deviation,
            "sample_policy": self._policy_kwargs["sample"],
            **super().planner_settings,
        }
        return deepcopy(settings)

    def plan(self, idx_action: int, env, mode="prod"):
        super().plan(idx_action, env, mode=mode)
        action = self._parallel_rollout(idx_action, env, self._branches)
        return action

    def _parallel_rollout(self, idx_action: int, env, branches):
        """Parallelize computation for faster trajectory simulation. Use this method
        for model-based forward prediction when the state evolution can be batched.
        """
        # Compute current V(s) or Q(s, a)
        use_learned_dynamics = self._use_learned_dynamics(idx_action)
        curr_state = env._get_observation()
        if use_learned_dynamics:
            curr_state = self._encode_state(idx_action, curr_state)
        actor_kwargs = {"envs": env, "states": curr_state}
        curr_actions = self._actor_interface(
            idx_action,
            samples=self._samples,
            variance=None
            if self._standard_deviation is None
            else self._standard_deviation**2,
            **actor_kwargs,
            **self._policy_kwargs
        )
        critic_kwargs = {"states": curr_state, "actions": curr_actions}
        curr_returns = self._critic_interface(idx_action, **critic_kwargs)

        # Simulate forward environments
        num = None if use_learned_dynamics else curr_actions.shape[0]
        if use_learned_dynamics:
            simulation_kwargs = {
                "states": curr_state,
                "actions": curr_actions,
            }
            # TODO: Preload envs to avoid overhead.
            curr_envs = self._clone_env(idx_action, env, num=num)
        else:
            curr_envs = self._clone_env(idx_action, env, num=num)
            simulation_kwargs = {
                "envs": curr_envs,
                "states": curr_state,
                "actions": curr_actions,
            }
        next_states, _ = self._simulate_interface(idx_action, **simulation_kwargs)

        # Rollout trajectories
        stack = [(idx_action, curr_envs, next_states, branches)]
        while stack:
            idx, curr_envs, next_states, branches = stack.pop()
            if not branches:
                continue

            var = branches.pop()
            to_stack = []
            if branches:
                to_stack.append((idx, curr_envs, next_states, branches))

            # Query Q(s, a) for optimization variable
            if isinstance(var, int):
                if not use_learned_dynamics:
                    next_envs = self._load_env(var, curr_envs)
                    next_states = np.array(
                        [next_env._get_observation() for next_env in next_envs]
                    )
                else:
                    # TODO: Preload envs to avoid overhead.
                    next_envs = self._load_env(var, curr_envs)

                actor_kwargs = {"envs": next_envs, "states": next_states}
                next_actions = self._actor_interface(var, **actor_kwargs)
                critic_kwargs = {"states": next_states, "actions": next_actions}
                next_returns = self._critic_interface(var, **critic_kwargs)
                curr_returns = getattr(np, self._mode)(
                    (curr_returns, next_returns), axis=0
                )

            # Simulate branches forward for simulation variable
            elif isinstance(var, dict):
                sim_idx, sim_branches = list(var.items())[0]

                if not use_learned_dynamics:
                    next_envs = self._load_env(sim_idx, curr_envs)
                    next_states = np.array(
                        [next_env._get_observation() for next_env in next_envs]
                    )
                else:
                    next_envs = self._load_env(sim_idx, curr_envs)

                actor_kwargs = {"envs": next_envs, "states": next_states}
                next_actions = self._actor_interface(sim_idx, **actor_kwargs)
                simulation_kwargs = {
                    "envs": next_envs,
                    "states": next_states,
                    "actions": next_actions,
                }
                next_next_states, _ = self._simulate_interface(
                    sim_idx, **simulation_kwargs
                )
                to_stack.append((sim_idx, next_envs, next_next_states, sim_branches))

            stack.extend(to_stack)

        best_action = curr_actions[curr_returns.argmax()]

        if isinstance(best_action, torch.Tensor):
            best_action = best_action.cpu().detach().numpy()

        return best_action
