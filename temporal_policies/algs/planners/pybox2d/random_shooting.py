import numpy as np

from .pybox2d_base import Box2DTrajOptim


class RandomShootingPlanner(Box2DTrajOptim):

    def __init__(self, samples, **kwargs):
        """Construct and rollout trajectories composed of actions randomly sampled from the action space of
        each primitive. Return the action with the highest scoring trajectory.

        args:
            samples: number of randomly sampled trajectories
        """
        super(RandomShootingPlanner, self).__init__(**kwargs)
        self._samples = samples

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
            next_env = self._clone_env(curr_env, opt_idx)
            next_state = next_env._get_observation()
            next_action = next_env.action_space.sample()
            next_q_val = self._q_function(opt_idx, next_state, next_action)
            q_val = getattr(np, self._mode)(q_val, next_q_val, axis=0)
        
        # Recursively simulate branches forward
        sim_vars = [x for x in branches if isinstance(x, dict)]
        for sim_dict in sim_vars:
            sim_idx, sim_branches = list(sim_dict.items())[0]
            next_env = self._clone_env(curr_env, sim_idx)
            next_q_val = self._random_rollout(next_env, sim_idx, sim_branches)
            q_val = getattr(np, self._mode)(q_val, next_q_val, axis=0)

        return action, q_val.item() if idx == self._idx else q_val 

    def _random_rollout(self, env, idx, branches):
        """Perform random trajectory rollouts on task structure as defined by branches.
        """

        # Compute current Q(s, a)
        state = env._get_observation()
        actions = np.array([env.action_space.sample() for _ in range(self._samples)])        
        q_vals = self._q_function(idx, state, actions)
        
        # Simulate forward environment
        curr_envs = self._clone_env(env, idx, num=self._samples) 
        curr_envs = [self._simulate_env(curr_env, action)[0] for curr_env, action in zip(curr_envs, actions)]
        
        # Rollout trajectories
        env_stack = [curr_envs]
        branches_stack = [branches]
        while branches_stack:

            branches = branches_stack.pop()


            # Query optimization variables
            opt_vars = [x for x in branches if isinstance(x, int)]
            for opt_idx in opt_vars:
                next_env = self._clone_env(curr_env, opt_idx)
                next_state = next_env._get_observation()
                next_action = next_env.action_space.sample()
                next_q_val = self._q_function(opt_idx, next_state, next_action)
                q_val = getattr(np, self._mode)(q_val, next_q_val, axis=0)
            
            # Recursively simulate branches forward
            sim_vars = [x for x in branches if isinstance(x, dict)]
            for sim_dict in sim_vars:
                sim_idx, sim_branches = list(sim_dict.items())[0]
                next_env = self._clone_env(curr_env, sim_idx)
                next_q_val = self._random_rollout(next_env, sim_idx, sim_branches)
                q_val = getattr(np, self._mode)(q_val, next_q_val, axis=0)

        return action, q_val.item() if idx == self._idx else q_val 

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        q_vals = np.zeros(self._samples, dtype=np.float32)
        actions = []
        for i in range(self._samples):
            action, q_vals[i] = self._random_rollout(env, idx, self._branches)
            actions.append(action)
        return actions[q_vals.argmax()]
