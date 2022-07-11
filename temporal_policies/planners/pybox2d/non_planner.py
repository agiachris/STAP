from .pybox2d_base import Box2DPlannerBase


class NonPlanner(Box2DPlannerBase):
    def __init__(self, **kwargs):
        """Reactive random or learned policy; no planning involved."""
        super().__init__(**kwargs)

    @property
    def planner_settings(self):
        return super().planner_settings

    def plan(self, idx, env, mode="prod"):
        super().plan(idx, env, mode=mode)
        actor_kwargs = {"states": env._get_observation(), "envs": env}
        action = self._actor_interface(idx, **actor_kwargs)
        return action
