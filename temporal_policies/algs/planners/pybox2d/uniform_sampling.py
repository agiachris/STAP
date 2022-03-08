import numpy as np

from .pybox2d_base import Box2DTrajOptim


class UniformSamplingPlanner(Box2DTrajOptim):

    def __init__(self, samples, agg_mode="max", **kwargs):
        """For each primitive, discretize the action space uniformly, rolling out each combination of actions
        in sequence and return the highest scoring trajectory. This method is computationally intractable in 
        practice, as its branching factor is exponential in the the task length.

        args:
            samples: nunber of uniform samples across the action space for each node in the planning graph
            agg_mode: value aggregation method at optimization nodes - hasattr(np, agg_mode) == True
        
        """
        super(UniformSamplingPlanner, self).__init__(**kwargs)
        self._samples = samples
        self._agg_mode = agg_mode

    def _q_over_action_space(self, env, idx, agg=True):
        state = env._get_observation()
        actions, _ = env._interp_actions(self._samples, self._task[idx]["dims"])
        q_vals = self._q_function(idx, state, actions)
        if agg: return getattr(np, self._agg_mode)(q_vals)
        return q_vals

    def _uniform_rollout(self, env, idx, branches):
        """Perform recursive uniform rollout on task structure as defined by branches.
        """
        # Compute current Q(s, a) 
        state = env._get_observation()
        actions, _ = env._interp_actions(self._samples, self._task[idx]["dims"])
        curr_qs += self._q_function(idx, state, actions)
        
        # Simulate forward environments
        mask = np.zeros(actions.shape[0], dtype=bool)
        curr_envs = self._clone_env(env, idx, num=self._samples)
        for i, (action, curr_env) in enumerate(zip(actions, curr_envs)):
            curr_env, mask[i] = self._simulate_env(curr_env, action)
        
        # Retain only successful actions
        actions = actions[mask]
        curr_qs = curr_qs[mask]
        curr_envs = [curr_envs[i] for i in range(len(curr_envs)) if mask[i]]

        # Query optimization variables
        opt_vars = [x for x in branches if isinstance(x, int)]
        for opt_idx in opt_vars:
            next_qs = np.zeros_like(curr_qs)
            for i, curr_env in enumerate(curr_envs):
                next_env = self._clone_env(curr_env, opt_idx)
                next_qs[i] = self._q_over_action_space(next_env, opt_idx, agg=True)
            curr_qs = getattr(np, self._mode)((curr_qs, next_qs), axis=0)
        
        # Recursively simulate branches forward
        sim_vars = [x for x in branches if isinstance(x, dict)]
        for sim_dict in sim_vars:
            sim_idx, sim_branches = list(sim_dict.items())[0]
            next_qs = np.zeros_like(curr_qs)
            for i, curr_env in enumerate(curr_envs):
                next_env = self._clone_env(curr_env, sim_idx)
                next_qs[i] = self._uniform_rollout(next_env, sim_idx, sim_branches)
            curr_qs = getattr(np, self._mode)((curr_qs, next_qs), axis=0)

        if idx != self._idx: return getattr(np, self._agg_mode)(curr_qs)
        return actions[curr_qs.argmax()]

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        action = self._uniform_rollout(env, idx, self._branches)
        return action
