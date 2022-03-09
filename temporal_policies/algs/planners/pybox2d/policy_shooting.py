import numpy as np

from .pybox2d_base import Box2DTrajOptim


class PolicyShootingPlanner(Box2DTrajOptim):

    def __init__(self, samples, variance=None, parallelize=True, sample_policy=False, **kwargs):
        """Construct and rollout trajectories composed of actions sampled from a multivariate Gaussian
        distribution centered at the policy's prediction. Return the action with the highest scoring trajectory.

        args: 
            samples: number of policy guided trajectories to sample
            variance: float or list of float of covariance of the diagonal Gaussian
            parallelize: whether or not to role out the trajectories in parallel
            sample_policy: whether or not stochastic policy outputs should be sampled or taken at the mean
        """
        super(PolicyShootingPlanner, self).__init__(**kwargs)
        self._samples = samples
        self._variance = variance
        self._parallelize = parallelize
        self._policy_kwargs = {"sample": sample_policy}

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        if self._parallelize: action = self._plan_parallel(env, idx)
        else: action = self._plan_sequential(env, idx)
        return action

    def _plan_sequential(self, env, idx):
        """Rollout trajectories one at a time. The efficiency of this method is approximately 
        equivalent to self.plan_parallel when using gym environments to forward simulate the state.
        """
        q_vals = np.zeros(self._samples, dtype=np.float32)
        actions = []
        for i in range(self._samples):
            action, q_vals[i] = self._policy_rollout(env, idx, self._branches)
            actions.append(action)
        return actions[q_vals.argmax()]

    def _plan_parallel(self, env, idx):
        """Parallelize computation for faster trajectory simulation. Use this method when 
        for model-based forward prediction for which state evolution can be batched.
        """
        action = self._parallel_policy_rollout(env, idx, self._branches)
        return action

    def _policy_rollout(self, env, idx, branches):
        """Perform policy-guided trajectory rollouts on task structure as defined by branches.
        """
        # Compute current Q(s, a)
        state = env._get_observation()
        variance = self._variance if self._idx == idx else None
        action = self._policy(idx, state, variance=variance, **self._policy_kwargs)
        q_val = self._q_function(idx, state, action)

        # Simulate forward environment
        curr_env = self._clone_env(env, idx)
        curr_env, _ = self._simulate_env(curr_env, action)

        # Query optimization variables
        opt_vars = [x for x in branches if isinstance(x, int)]
        for opt_idx in opt_vars:
            next_env = self._load_env(curr_env, opt_idx)
            next_state = next_env._get_observation()
            next_action = self._policy(opt_idx, next_state, **self._policy_kwargs)
            next_q_val = self._q_function(opt_idx, next_state, next_action)
            q_val = getattr(np, self._mode)((q_val, next_q_val), axis=0)
        
        # Recursively simulate branches forward
        sim_vars = [x for x in branches if isinstance(x, dict)]
        for sim_dict in sim_vars:
            sim_idx, sim_branches = list(sim_dict.items())[0]
            next_env = self._load_env(curr_env, sim_idx)
            next_q_val = self._policy_rollout(next_env, sim_idx, sim_branches)
            q_val = getattr(np, self._mode)((q_val, next_q_val), axis=0)

        return action, q_val.item() if idx == self._idx else q_val 

    def _parallel_policy_rollout(self, env, idx, branches):
        """Perform policy-gudied trajectory rollouts on task structure as defined by branches.
        """
        # Compute current Q(s, a)
        state = env._get_observation()
        primitive_actions = self._policy(idx, state, samples=self._samples, variance=self._variance, **self._policy_kwargs)
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
                actions = self._policy(var, states, **self._policy_kwargs)
                next_q_vals = self._q_function(var, states, actions)
                q_vals = getattr(np, self._mode)((q_vals, next_q_vals), axis=0)

            # Simulate branches forward for simulation variable
            elif isinstance(var, dict):
                sim_idx, sim_branches = list(var.items())[0]
                next_envs = [self._load_env(curr_env, sim_idx) for curr_env in curr_envs]
                states = np.array([next_env._get_observation() for next_env in next_envs])
                actions = self._policy(sim_idx, states, **self._policy_kwargs)
                next_envs = [self._simulate_env(next_env, action) for next_env, action in zip(next_envs, actions)]
                to_stack.append((sim_branches, next_envs, sim_idx))
            
            stack.extend(to_stack)

        return primitive_actions[q_vals.argmax()]
