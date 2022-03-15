
import numpy as np

from .pybox2d_base import Box2DPlannerBase


class ShootingPlanner(Box2DPlannerBase):

    def __init__(self, samples, standard_deviation=None, sample_policy=False, **kwargs):
        """Perform shooting-based model-predictive control with randomly sampled actions OR actions
        sampled from a multivariate Gaussian distribution centered at the policy's prediction. Return
        the first action of the highest scoring trajectory.

        args:
            samples: number of sampled trajectories
            standard_deviation: float standard deviation parameterizing diagonal Gaussian
            sample_policy: whether or not stochastic policy outputs should be sampled or taken at the mean
        """
        super().__init__(**kwargs)
        self._samples = samples
        self._variance = standard_deviation ** 2
        self._policy_kwargs = {"sample": sample_policy}
    
    def plan(self, idx, env, mode="prod"):
        super().plan(idx, env, mode=mode)
        action = self._parallel_rollout(idx, env, self._branches)
        return action
    
    def _parallel_rollout(self, idx, env, branches):
        """Parallelize computation for faster trajectory simulation. Use this method 
        for model-based forward prediction when the state evolution can be batched.
        """
        # Compute current V(s) or Q(s, a)
        use_learned_dynamics = self._use_learned_dynamics(idx)
        curr_state = env._get_observation()
        if use_learned_dynamics: curr_state = self._encode_state(idx, curr_state)
        actor_kwargs = {"envs": env, "states": curr_state}
        curr_actions = self._actor_interface(
            idx, samples=self._samples, variance=self._variance, 
            **actor_kwargs, **self._policy_kwargs
        )
        critic_kwargs = {"states": curr_state, "actions": curr_actions}
        curr_returns = self._critic_interface(idx, **critic_kwargs)
        
        # Simulate forward environments
        num = None if use_learned_dynamics else curr_actions.shape[0]
        curr_envs = self._clone_env(idx, env, num=num)
        simulation_kwargs = {"envs": curr_envs, "states": curr_state, "actions": curr_actions}
        next_states, _ = self._simulate_interface(idx, **simulation_kwargs)
        
        # Rollout trajectories
        stack = [(idx, curr_envs, next_states, branches)]
        while stack:
            idx, curr_envs, next_states, branches = stack.pop()
            if not branches: continue

            var = branches.pop()
            to_stack = []
            if branches: to_stack.append((idx, curr_envs, next_states, branches))

            # Query Q(s, a) for optimization variable
            if isinstance(var, int):
                if not use_learned_dynamics: 
                    next_envs = self._load_env(var, curr_envs)
                    next_states = np.array([next_env._get_observation() for next_env in next_envs])
                else: next_envs = [self._load_env(var, curr_envs)] * len(next_states)
                
                actor_kwargs = {"envs": next_envs, "states": next_states}
                next_actions = self._actor_interface(var, **actor_kwargs)
                critic_kwargs = {"states": next_states, "actions": next_actions}
                next_returns = self._critic_interface(var, **critic_kwargs)
                curr_returns = getattr(np, self._mode)((curr_returns, next_returns), axis=0)

            # Simulate branches forward for simulation variable
            elif isinstance(var, dict):
                sim_idx, sim_branches = list(var.items())[0]

                if not use_learned_dynamics: 
                    next_envs = self._load_env(sim_idx, curr_envs)
                    next_states = np.array([next_env._get_observation() for next_env in next_envs])
                else: next_envs = [self._load_env(sim_idx, curr_envs)] * len(next_states)

                actor_kwargs = {"envs": next_envs, "states": next_states}
                next_actions = self._actor_interface(sim_idx, **actor_kwargs)
                simulation_kwargs = {"envs": next_envs, "states": next_states, "actions": next_actions}
                next_next_states, _ = self._simulate_interface(sim_idx, **simulation_kwargs)
                if use_learned_dynamics: next_envs = next_envs[0]
                to_stack.append((sim_idx, next_envs, next_next_states, sim_branches))
            
            stack.extend(to_stack)

        return curr_actions[curr_returns.argmax()]
