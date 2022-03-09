import torch
import numpy as np

from .pybox2d_base import Box2DTrajOptim


class CEMRandomPlanner(Box2DTrajOptim):

    def __init__(self, samples, iterations=1, variance=None, **kwargs):
        """Perform the Cross-Entropy Method. Equivalent to random shooting and iteratively updating 
        and improving the sampling distribution over the specified number of iterations.

        args:
            samples: number of randomly sampled trajectories per iteration
            iterations: number of resampling iterations (i.e., updates to the sampling distribution)
            variance: float or list of float of covariance of the diagonal Gaussian at the first iteration
        """
        super(CEMRandomPlanner, self).__init__(**kwargs)

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        action = None
        return action

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        if self._parallelize: action = self._plan_parallel(env, idx)
        else: action = self._plan_sequential(env, idx)
        return action

    def _plan_parallel(self, env, idx):
        """Parallelize computation for faster trajectory simulation. Use this method when 
        for model-based forward prediction for which state evolution can be batched.
        """
        action = self._parallel_random_rollout(env, idx, self._branches)
        return action

    def _random_rollout(self, env, idx, branches):
        """Perform random trajectory rollouts on task structure as defined by branches.
        """
        # Compute current Q(s, a)
        state = env._get_observation()
        action = env.action_space.sample()
        q_val = self._q_function(idx, state, action)

        # Simulate forward environment
        curr_env = self._clone_env(env, idx)
        curr_env, _ = self._simulate_env(curr_env, action)

        # Query optimization variables
        opt_vars = [x for x in branches if isinstance(x, int)]
        for opt_idx in opt_vars:
            next_env = self._load_env(curr_env, opt_idx)
            next_state = next_env._get_observation()
            next_action = next_env.action_space.sample()
            next_q_val = self._q_function(opt_idx, next_state, next_action)
            q_val = getattr(np, self._mode)((q_val, next_q_val), axis=0)
        
        # Recursively simulate branches forward
        sim_vars = [x for x in branches if isinstance(x, dict)]
        for sim_dict in sim_vars:
            sim_idx, sim_branches = list(sim_dict.items())[0]
            next_env = self._load_env(curr_env, sim_idx)
            next_q_val = self._random_rollout(next_env, sim_idx, sim_branches)
            q_val = getattr(np, self._mode)((q_val, next_q_val), axis=0)

        return action, q_val.item() if idx == self._idx else q_val 

    def _parallel_random_rollout(self, env, idx, branches):
        """Perform random trajectory rollouts on task structure as defined by branches.
        """
        # Compute current Q(s, a)
        state = env._get_observation()
        primitive_actions = np.array([env.action_space.sample() for _ in range(self._samples)])
        q_vals = self._q_function(idx, state, primitive_actions)
        
        # Simulate forward environments
        curr_envs = self._clone_env(env, idx, num=self._samples) 
        curr_envs = [self._simulate_env(curr_env, action)[0] for curr_env, action in zip(curr_envs, primitive_actions)]
        
        # Rollout trajectories
        stack = [(branches, curr_envs, idx)]
        while stack:
            branches, curr_envs, idx = stack.pop()
            if not branches: continue

            var = branches.pop()
            to_stack = []
            if branches: to_stack.append((branches, curr_envs, idx))

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
                states = np.array([next_env._get_observation() for next_env in next_envs])
                actions = np.array([next_env.action_space.sample() for next_env in next_envs])
                next_envs = [self._simulate_env(next_env, action) for next_env, action in zip(next_envs, actions)]
                to_stack.append((sim_branches, next_envs, sim_idx))
            
            stack.extend(to_stack)

        return primitive_actions[q_vals.argmax()]