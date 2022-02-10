import torch
import numpy as np

from temporal_policies.utils import utils


class SamplingOptim:

    def __init__(self, models, configs):
        self._models = models

        self._configs = [config["env_kwargs"] for config in configs]
        self._env_cls = [type(model.env.unwrapped) for model in models]
        self._fmt = models[0]._format_batch

    def random_sampling(self, curr_env, num, mode="prod"):
        assert len(self._models) == 2
        curr_model, next_model = self._models
        curr_env_cls, next_env_cls = self._env_cls
        curr_config, next_config = self._configs

        # Compute current Q(s, a)
        states = np.tile(curr_env._get_observation(), (num, 1))
        actions = np.array([curr_env.action_space.sample() for _ in range(num)])
        curr_q1s, curr_q2s = curr_model.network.critic(self._fmt(states), self._fmt(actions))
        curr_qs = utils.to_np(torch.min(curr_q1s, curr_q2s))

        # Simulate forward environments
        curr_envs = [curr_env_cls.clone(curr_env, **curr_config) for _ in range(num)]
        next_envs = []
        for action, env in zip(actions, curr_envs):
            obs, rew, done, info = env.step(action)
            next_envs.append(next_env_cls.load(env, **next_config))
        
        # Compute next Q(s, a)
        next_qs = np.array(list(zip(*[env.action_value(next_model) for env in next_envs]))[0])

        qs = getattr(np, mode)((curr_qs, next_qs), axis=0)
        assert qs.shape == (num,) and qs.ndim == 1

        action = actions[qs.argmax(), :]
        return action
    
    def random_shooting(self, curr_env, num, var_x=0.1, var_theta=0.5, mode="prod"):
        assert len(self._models) == 2
        curr_model, next_model = self._models
        curr_env_cls, next_env_cls = self._env_cls
        curr_config, next_config = self._configs

        # Compute current Q(s, a)
        state = curr_env._get_observation()
        states = np.tile(state.copy(), (num, 1))
        actions = np.random.multivariate_normal(
            mean=curr_model.predict(state),
            cov=np.array([[var_x, 0], [0, var_theta]]),
            size=num
        )
        actions = np.clip(actions, -1, 1).astype(np.float32)
        assert actions.shape == (num, curr_env.action_space.shape[0])
        curr_q1s, curr_q2s = curr_model.network.critic(self._fmt(states), self._fmt(actions))
        curr_qs = utils.to_np(torch.min(curr_q1s, curr_q2s))

        # Simulate forward environments
        curr_envs = [curr_env_cls.clone(curr_env, **curr_config) for _ in range(num)]
        next_envs = []
        for action, env in zip(actions, curr_envs):
            obs, rew, done, info = env.step(action)
            next_envs.append(next_env_cls.load(env, **next_config))
        
        # Compute next Q(s, a)
        next_qs = np.array(list(zip(*[env.action_value(next_model) for env in next_envs]))[0])

        qs = getattr(np, mode)((curr_qs, next_qs), axis=0)
        assert qs.shape == (num,) and qs.ndim == 1

        action = actions[qs.argmax(), :]
        return action

    def cross_entropy(self, curr_env, num):
        assert len(self._models) == 2
        pass
