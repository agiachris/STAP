import torch
import numpy as np
from numpy.random import multivariate_normal

from .pybox2d_base import Box2DTrajOptim


class CEMRandomPlanner(Box2DTrajOptim):

    def __init__(self, samples, iterations=1, elites=1, variance=None, **kwargs):
        """Perform the Cross-Entropy Method. Equivalent to random shooting and iteratively updating 
        and improving the sampling distribution over the specified number of iterations.

        args:
            samples: number of randomly sampled trajectories per iteration
            iterations: number of resampling iterations (i.e., updates to the sampling distribution)
            elites: float percentage of trajectories to retain ordered by rewards at each iteration
            variance: float or list of float of covariance of the diagonal Gaussian at the first iteration
        """
        super(CEMRandomPlanner, self).__init__(**kwargs)
        self._samples = samples
        self._iterations = iterations
        self._elites = elites
        assert 0 < self._elites <= 1, "Elites must be a percentage between (0, 1]"
        self._top_k = np.round(self._samples * self._elites).astype(int).item()
        self._variance = variance

    def plan(self, env, idx, mode="prod"):
        """Parallelize computation for faster trajectory simulation. Use this method when 
        for model-based forward prediction for which state evolution can be batched.
        """
        super().plan(env, idx, mode=mode)
        action = self._incremental_cem(env, idx)
        return action

    def _incremental_cem(self, env, idx):
        """Plan via incrementally refine random trajectory sampling distribution.
        """
        # Initialize sampling distribution
        actions = np.array([env.action_space.sample() for _ in range(self._samples**2)])
        mean = actions.mean(0)
        cov = np.eye(actions.shape[1]) @ actions.var(0)
        
        # Refine distribution incrementally
        for _ in range(self._iterations):
            
            # Compute current Q(s, a)
            state = env._get_observation()
            actions = multivariate_normal(mean, cov, size=self._samples)
            q_vals = self._q_function(idx, state, actions)
            
            # Simulate forward environments
            curr_envs = self._clone_env(env, idx, num=self._samples) 
            for curr_env, action in zip(curr_envs, actions): self._simulate_env(curr_env, action)

            # Rollout trajectories
            traj_q_vals = self._parallel_random_rollout(curr_envs, self._branches)
            q_vals = getattr(np, self._mode)((q_vals, traj_q_vals), axis=0)

            # Update sampling distribution
            rank_order = q_vals.argsort()
            q_vals = q_vals[rank_order]
            actions = actions[rank_order]

            if self._elites == 1:
                # Weight distribution by returns
                continue
            
            actions = actions[self._top_k]
            mean = actions.mean(0)
            cov = actions.var(0)
                

    def _parallel_random_rollout(self, curr_envs, branches):
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
