import numpy as np
from copy import deepcopy

import torch

from .pybox2d_base import Box2DPlannerBase


class CrossEntropyMethod(Box2DPlannerBase):
    def __init__(
        self,
        samples,
        elites,
        iterations,
        standard_deviation,
        momentum=0.0,
        population_decay=1.0,
        best_action=True,
        keep_elites_fraction=0.0,
        sample_policy=False,
        **kwargs
    ):
        """Improved Cross-Entropy Method implementation supporting both trajectories constructed
        from random actions and policy centered actions.

        args:
            samples: number of randomly sampled trajectories per CEM-iteration
            elites: number of elites per CEM-iteration
            iterations: number of CEM-iterations per environment step
            standard_deviation: float standard deviation parameterizing diagonal Gaussian
            momentum: weighted average of sampling distribution mean per CEM-iteration - float [0, 1]
            population_decay: geometric decay rate of population - float [0, 1]
            best_action: if True, returns the first action of the best sampled trajectory
                         if False, return the first action of the sampling distribution mean
            keep_elites_fraction: fraction of elites to store in population for next CEM-iteration
            sample_policy: whether or not stochastic policy outputs should be sampled or taken at the mean
        """
        super().__init__(**kwargs)
        self._samples = samples
        self._elites = elites
        self._iterations = iterations
        self._standard_deviation = standard_deviation
        self._momentum = momentum
        self._population_decay = population_decay
        self._best_action = best_action
        self._keep_elites_fraction = keep_elites_fraction
        self._policy_kwargs = {"sample": sample_policy}

    @property
    def planner_settings(self):
        settings = {
            "samples": self._samples,
            "elites": self._elites,
            "iterations": self._iterations,
            "standard_deviation": self._standard_deviation,
            "momentum": self._momentum,
            "population_decay": self._population_decay,
            "best_action": self._best_action,
            "keep_elites_fraction": self._keep_elites_fraction,
            "sample_policy": self._policy_kwargs["sample"],
            **super().planner_settings,
        }
        return deepcopy(settings)

    def plan(self, idx, env, mode="prod"):
        super().plan(idx, env, mode=mode)
        self._init_cem_params()
        action = self._incremental_cem(idx, env)
        return action

    def _init_mean(self, relative=True):
        if self._use_policy_actor(self._idx):
            state = self._env._get_observation()
            if self._use_learned_dynamics(self._idx):
                state = self._encode_state(self._idx, state)
            return self._policy(self._idx, state, **self._policy_kwargs)
        low = self._env.action_space.low
        high = self._env.action_space.high
        if relative:
            return (high + low) / 2.0
        return np.zeros_like(low, dtype=np.float32)

    def _init_std(self, relative=True):
        low = self._env.action_space.low
        high = self._env.action_space.high
        if relative:
            return (high - low) / 2.0 * self._standard_deviation
        return np.ones_like(low, dtype=np.float32) * self._standard_deviation

    def _init_cem_params(self):
        """Initialize CEM parameters."""
        self._mean = self._init_mean(True)
        self._std = self._init_std(True)
        self._population = self._samples
        self._prev_mean = None
        self._prev_elites = None
        self._optimal_action = None
        self._optimal_return = float("-inf")

    def _update_cem_params(self, actions, returns):
        """Update CEM parameters.
        args:
            actions: np.array of sampled actions
            returns: np.array of returns under actions
        """
        # Compute elites
        num_elites = max(2, min(self._elites, self._population // 2))
        num_elites = int(round(num_elites))
        actions = actions[returns.argsort()]
        elites = actions[:num_elites]

        # Retain fraction of elites for next CEM-iteration
        keep_elites = int(round(num_elites * self._keep_elites_fraction))
        self._prev_elites = elites[:keep_elites] if keep_elites >= 1 else None

        # Update mean and std
        self._prev_mean = self._mean.copy()
        self._mean = self._momentum * self._mean + (1 - self._momentum) * elites.mean(0)
        self._std = self._momentum * self._std + (1 - self._momentum) * elites.std(0)

        # Update population size
        self._population = max(
            self._population * self._population_decay, 2 * self._elites
        )
        self._population = int(round(self._population))

    def _sample_actions(self):
        """Sample actions according to current mean and variance. Append
        mean and subset of elites from previous CEM-iteration.
        """
        low = self._env.action_space.low
        high = self._env.action_space.high
        actions = self._mean + self._std * np.random.randn(self._population, *low.shape)
        if self._prev_mean is not None:
            actions = np.concatenate(
                (actions, np.expand_dims(self._prev_mean, 0)), axis=0
            )
        if self._prev_elites is not None:
            actions = np.concatenate((actions, self._prev_elites), axis=0)
        return np.clip(actions, low, high).astype(np.float32)

    def _incremental_cem(self, idx, env):
        """Cross-Entropy Method outer-loop."""
        for _ in range(self._iterations):
            actions, returns = self._parallel_rollout(idx, env, self._branches)

            if returns.max().item() > self._optimal_return:
                self._optimal_action = actions[returns.argmax()]
                self._optimal_return = returns.max().item()

            self._update_cem_params(actions, returns)

        return self._optimal_action if self._best_action else self._mean

    def _parallel_rollout(self, idx_action: int, env, branches):
        """Parallelize computation for faster trajectory simulation. Use this method
        for model-based forward prediction when the state evolution can be batched.
        """
        # Compute current V(s) or Q(s, a)
        use_learned_dynamics = self._use_learned_dynamics(idx_action)
        curr_state = env._get_observation()
        if use_learned_dynamics:
            curr_state = self._encode_state(idx_action, curr_state)
        curr_actions = self._sample_actions()
        critic_kwargs = {"states": curr_state, "actions": curr_actions}
        curr_returns = self._critic_interface(idx_action, **critic_kwargs)

        # Simulate forward environments
        num = None if use_learned_dynamics else curr_actions.shape[0]
        if use_learned_dynamics:
            simulation_kwargs = {
                "states": curr_state,
                "actions": curr_actions,
            }
            # TODO: Preload envs to avoid overhead.
            curr_envs = self._clone_env(idx_action, env, num=num)
        else:
            curr_envs = self._clone_env(idx_action, env, num=num)
            simulation_kwargs = {
                "envs": curr_envs,
                "states": curr_state,
                "actions": curr_actions,
            }
        next_states, _ = self._simulate_interface(idx_action, **simulation_kwargs)

        # Rollout trajectories
        stack = [(idx_action, curr_envs, next_states, branches)]
        while stack:
            idx, curr_envs, next_states, branches = stack.pop()
            if not branches:
                continue

            var = branches.pop()
            to_stack = []
            if branches:
                to_stack.append((idx, curr_envs, next_states, branches))

            # Query Q(s, a) for optimization variable
            if isinstance(var, int):
                if not use_learned_dynamics:
                    next_envs = self._load_env(var, curr_envs)
                    next_states = np.array(
                        [next_env._get_observation() for next_env in next_envs]
                    )
                else:
                    # TODO: Preload envs to avoid overhead.
                    next_envs = self._load_env(var, curr_envs)

                actor_kwargs = {"envs": next_envs, "states": next_states}
                next_actions = self._actor_interface(var, **actor_kwargs)
                critic_kwargs = {"states": next_states, "actions": next_actions}
                next_returns = self._critic_interface(var, **critic_kwargs)
                curr_returns = getattr(np, self._mode)(
                    (curr_returns, next_returns), axis=0
                )

            # Simulate branches forward for simulation variable
            elif isinstance(var, dict):
                sim_idx, sim_branches = list(var.items())[0]

                if not use_learned_dynamics:
                    next_envs = self._load_env(sim_idx, curr_envs)
                    next_states = np.array(
                        [next_env._get_observation() for next_env in next_envs]
                    )
                else:
                    next_envs = [self._load_env(sim_idx, curr_envs)] * len(next_states)

                actor_kwargs = {"envs": next_envs, "states": next_states}
                next_actions = self._actor_interface(sim_idx, **actor_kwargs)
                simulation_kwargs = {
                    "envs": next_envs,
                    "states": next_states,
                    "actions": next_actions,
                }
                next_next_states, _ = self._simulate_interface(
                    sim_idx, **simulation_kwargs
                )
                to_stack.append((sim_idx, next_envs, next_next_states, sim_branches))

            stack.extend(to_stack)

        if isinstance(curr_actions, torch.Tensor):
            curr_actions = curr_actions.cpu().detach().numpy()
        if isinstance(curr_returns, torch.Tensor):
            curr_returns = curr_returns.cpu().detach().numpy()

        return curr_actions, curr_returns
