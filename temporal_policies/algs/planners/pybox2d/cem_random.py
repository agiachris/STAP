import torch
import numpy as np

from .cem_base import CrossEntropyMethod


class CEMRandomPlanner(CrossEntropyMethod):

    def __init__(self, **kwargs):
        """Cross-Entropy Method with randomly sampled actions. Equivalent to RandomShootingPlanner
        while adapting the sampling distribution of actions over several iterations to maximize returns.
        """
        super(CEMRandomPlanner, self).__init__(**kwargs)

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        action = self._incremental_cem(env, idx)
        return action

    def _parallel_rollout(self, curr_envs, branches):
        """Perform random trajectory rollouts on task structure as defined by branches.
        """
        if self._mode == "prod": q_vals = np.ones(self._samples)
        elif self._mode == "sum": q_vals = np.zeros(self._samples)
        else: raise ValueError(f"Value aggregation mode {self._mode} is unsupported")

        stack = [(branches, curr_envs)]
        while stack:
            branches, curr_envs = stack.pop()
            if not branches: continue

            var = branches.pop()
            to_stack = []
            if branches: to_stack.append((branches, curr_envs))

            # Query Q(s, a) for optimization variable
            if isinstance(var, int):
                next_envs = [self._load_env(curr_env, var) for curr_env in curr_envs]
                states = np.array([next_env._get_observation() for next_env in next_envs])
                actions = np.array([next_env.action_space.sample() for next_env in next_envs])
                next_q_vals = self._q_function(var, states, actions)
                q_vals = getattr(np, self._mode)((q_vals, next_q_vals), axis=0)

            # Simulate branches forward for simulation variable
            elif isinstance(var, dict):
                sim_idx, sim_branches = list(var.items())[0]
                next_envs = [self._load_env(curr_env, sim_idx) for curr_env in curr_envs]
                actions = np.array([next_env.action_space.sample() for next_env in next_envs])
                for next_env, action in zip(next_envs, actions): self._simulate_env(next_env, action)
                to_stack.append((sim_branches, next_envs))
            
            stack.extend(to_stack)

        return q_vals
