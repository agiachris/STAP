import torch
import numpy as np

from .pybox2d_base import Box2DTrajOptim


class PolicyShootingPlanner(Box2DTrajOptim):

    def __init__(self, samples, variance, sample_downstream=False, **kwargs):
        """Construct and rollout trajectories composed of actions sampled from a multivariate Gaussian
        distribution centered at the policy's prediction. Return the action with the highest scoring trajectory.

        args: 
            samples: number of policy guided trajectories to sample
            variance: float or list of float of covariance of the diagonal Gaussian
            sample_downstream: if False, only the first primitive (being optimized) will be sampled from a 
                               multivariate Gaussian distribution, while other primitives will be deterministically 
                               queried from the policy. If False, all primitives downstream primitives are sampled.
        """
        super(PolicyShootingPlanner, self).__init__(**kwargs)
        self._samples = samples
        self._variance = variance
        self._sample_downstream = sample_downstream

    def _policy_rollout(self, env, idx, branches):
        """Perform random trajectory rollouts on task structure as defined by branches.
        """
        # Compute current Q(s, a)
        action = env.action_space.sample()
        state = env._get_observation()
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

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        q_vals = np.zeros(self._samples, dtype=np.float32)
        actions = []
        for i in range(self._samples):
            action, q_vals[i] = self._random_rollout(env, idx, self._branches)
            actions.append(action)
        return actions[q_vals.argmax()]

    def plan(self, env, idx, samples, mode="prod"):
       
        super().plan(env, idx, mode=mode)
        
        # assert len(self._models) == 2
        # curr_model, next_model = self._models
        # curr_env_cls, next_env_cls = self._env_cls
        # curr_config, next_config = self._configs

        # # Compute current Q(s, a)
        # state = curr_env._get_observation()
        # states = np.tile(state.copy(), (samples, 1))
        # actions = np.random.multivariate_normal(
        #     mean=curr_model.predict(state),
        #     cov=np.array([[var_x, 0], [0, var_theta]]),
        #     size=samples
        # )
        # actions = np.clip(actions, -1, 1).astype(np.float32)
        # assert actions.shape == (samples, curr_env.action_space.shape[0])
        # curr_q1s, curr_q2s = curr_model.network.critic(self._fmt(states), self._fmt(actions))
        # curr_qs = utils.to_np(torch.min(curr_q1s, curr_q2s))

        # # Simulate forward environments
        # curr_envs = [curr_env_cls.clone(curr_env, **curr_config) for _ in range(samples)]
        # next_envs = []
        # for action, env in zip(actions, curr_envs):
        #     obs, rew, done, info = env.step(action)
        #     next_envs.append(next_env_cls.load(env, **next_config))
        
        # # Compute next Q(s, a)
        # next_qs = np.array(list(zip(*[env.action_value(next_model) for env in next_envs]))[0])

        # qs = getattr(np, mode)((curr_qs, next_qs), axis=0)
        # assert qs.shape == (samples,) and qs.ndim == 1

        # action = actions[qs.argmax(), :]
        # return action

        action = None
        return action
