from .pybox2d_base import Box2DTrajOptim


class RandomPolicy(Box2DTrajOptim):

    def __init__(self, **kwargs):
        super(RandomPolicy, self).__init__(**kwargs)

    def plan(self, env, idx, mode="prod"):
        """Simply sample the action space; no planning.
        """
        super().plan(env, idx, mode=mode)
        action = env.action_space.sample()
        return action
