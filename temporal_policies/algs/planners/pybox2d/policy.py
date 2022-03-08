from .pybox2d_base import Box2DTrajOptim


class Policy(Box2DTrajOptim):

    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)

    def plan(self, env, idx, mode="prod"):
        """Simply evaluate the policy; no planning.
        """
        super().plan(env, idx, mode=mode)
        obs = env._get_observation()
        action = self._get_model(idx).predict(obs)
        return action
