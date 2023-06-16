import numpy as np
from copy import deepcopy

from .pybox2d_base import Box2DPlannerBase


class UniformSamplingPlanner(Box2DPlannerBase):
    def __init__(self, samples, agg_mode="max", **kwargs):
        """For each primitive, discretize the action space uniformly, rolling out each combination of actions
        in sequence and return the highest scoring trajectory. This method is computationally intractable in
        practice, as its branching factor is exponential in the the task length.

        args:
            samples: nunber of uniform samples across the action space for each node in the planning graph
            agg_mode: value aggregation method at optimization nodes - hasattr(np, agg_mode) == True
        """
        super().__init__(**kwargs)
        assert hasattr(np, agg_mode)
        self._samples = samples
        self._agg_mode = agg_mode

    @property
    def planner_settings(self):
        settings = {
            "samples": self._samples,
            "agg_mode": self._agg_mode,
            **super().planner_settings,
        }
        return deepcopy(settings)

    def plan(self, idx, env, mode="prod"):
        super().plan(idx, env, mode=mode)
        action = self._uniform_rollout(idx, env, self._branches)
        return action

    def _uniform_rollout(self, idx, env, branches, state=None):
        """Perform recursive uniform rollout on task structure as defined by branches."""
        # Compute current V(s) or Q(s, a)
        use_learned_dynamics = self._use_learned_dynamics(idx)
        curr_state = env._get_observation()
        if use_learned_dynamics:
            if state is None:
                curr_state = self._encode_state(idx, curr_state)
            else:
                curr_state = state
        curr_actions, _ = env._interp_actions(self._samples, self._task[idx]["dims"])
        critic_kwargs = {"states": curr_state, "actions": curr_actions}
        curr_returns = self._critic_interface(idx, **critic_kwargs)

        # Simulate forward environments
        num = None if use_learned_dynamics else curr_actions.shape[0]
        curr_envs = self._clone_env(idx, env, num=num)
        simulation_kwargs = {
            "envs": curr_envs,
            "states": curr_state,
            "actions": curr_actions,
        }
        next_states, success = self._simulate_interface(idx, **simulation_kwargs)

        # No valid action, simply return the highest scoring one
        if not success.max():
            if idx != self._idx:
                return getattr(np, self._agg_mode)(curr_returns)
            return curr_actions[curr_returns.argmax()]

        # Retain only successful branches
        if not use_learned_dynamics:
            curr_actions = curr_actions[success]
            curr_returns = curr_returns[success]
            next_states = next_states[success]
            curr_envs = [curr_envs[i] for i in range(len(curr_envs)) if success[i]]
        else:
            success.sum() == len(success)

        # Query optimization variables
        opt_vars = [x for x in branches if isinstance(x, int)]
        for opt_idx in opt_vars:
            next_returns = np.zeros_like(curr_returns)
            if not use_learned_dynamics:
                next_envs = self._load_env(opt_idx, curr_envs)
            else:
                next_envs = [self._load_env(opt_idx, curr_envs)] * len(success)
            for i, (next_env, next_state) in enumerate(zip(next_envs, next_states)):
                next_returns[i] = self._critic_over_action_space(
                    opt_idx, next_env, state=next_state, agg=True
                )
            curr_returns = getattr(np, self._mode)((curr_returns, next_returns), axis=0)

        # Recursively simulate branches forward
        sim_vars = [x for x in branches if isinstance(x, dict)]
        for sim_dict in sim_vars:
            sim_idx, sim_branches = list(sim_dict.items())[0]
            next_returns = np.zeros_like(curr_returns)
            if not use_learned_dynamics:
                next_envs = self._load_env(sim_idx, curr_envs)
            else:
                next_envs = [self._load_env(sim_idx, curr_envs)] * len(success)
            for i, (next_env, next_state) in enumerate(zip(next_envs, next_states)):
                next_returns[i] = self._uniform_rollout(
                    sim_idx, next_env, sim_branches, state=next_state
                )
            curr_returns = getattr(np, self._mode)((curr_returns, next_returns), axis=0)

        if idx != self._idx:
            return getattr(np, self._agg_mode)(curr_returns).item()
        return curr_actions[curr_returns.argmax()]

    def _critic_over_action_space(self, idx, env, state=None, agg=True):
        # Compute current V(s) or Q(s, a)
        curr_state = env._get_observation()
        if self._use_learned_dynamics(idx):
            if state is None:
                curr_state = self._encode_state(idx, curr_state)
            else:
                curr_state = state
        curr_actions, _ = env._interp_actions(self._samples, self._task[idx]["dims"])
        critic_kwargs = {"states": curr_state, "actions": curr_actions}
        returns = self._critic_interface(idx, **critic_kwargs)
        if agg:
            returns = getattr(np, self._agg_mode)(returns)
        return returns
