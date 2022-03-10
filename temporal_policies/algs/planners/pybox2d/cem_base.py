import torch
import numpy as np
from numpy.random import multivariate_normal
from abc import ABC, abstractmethod

from .pybox2d_base import Box2DTrajOptim


class CrossEntropyMethod(Box2DTrajOptim, ABC):

    def __init__(self,
                 samples,
                 elites,
                 iterations,
                 momentum=0.0,
                 decay=0.0,
                 best_action=True,
                 keep_elites=0.0,
                 **kwargs):
        """Abstract base class defining skeleton for a modified Cross-Entropy Method implementation.

        args:
            samples: number of randomly sampled trajectories per CEM-iteration
            elites: number of elites per CEM-iteration
            iterations: number of CEM-iterations per environment step
            momentum: weighted average of sampling distribution mean per CEM-iteration - float [0, 1]
            decay: geometric decay rate of population - float [0, 1]
            best_action: if True, returns the first action of the best sampled trajectory
                         if False, return the first action of the sampling distribution mean
            keep_elites: fraction of elites to store in population for next CEM-iteration
        """
        Box2DTrajOptim.__init__(self, **kwargs)
        self._samples = samples
        self._elites = elites
        self._iterations = iterations
        self._momentum = momentum
        self._decay = decay
        self._best_action = best_action
        self._keep_elites = keep_elites

    def _init_cem(self):
        self._steps = 0
        self._mean = self._init_mean() 
        self._cov = self._init_variance()
    
    @abstractmethod
    def _init_mean(self):
        """Initialize mean for the first CEM-iteration.
        """
        pass

    @abstractmethod
    def _init_variance(self):
        """Initialize variance for the first CEM-iteration.
        """
        pass
    
    def _sample_actions(self, env):
        """Sample actions according to current mean and variance.
        """
        decayed_pop = round(self._samples * self._decay ** self._steps)
        num_samples = int(max(decayed_pop, 2 * self._elites))
        actions = multivariate_normal(self._mean, self._cov, num_samples)
        actions = np.clip(actions, env.action_space.low, env.action_space.high)
        return actions

    def _update_distribution(self, elites):
        """Update sampling distribution.
        """
        self._mean = self._momentum * self._mean + (1 - self._momentum) * elites.mean(0)
        self._cov = None

        pass

    def _incremental_cem(self, env, idx):
        """Cross-Entropy Method outer-loop.
        """
        self._init_cem()
        init_actions = None
        for self._steps in range(self._iterations):

            # Compute current Q(s, a)
            state = env._get_observation()
            if init_actions is not None: actions = np.concatenate((actions, init_actions), axis=0)
            q_vals = self._q_function(idx, state, actions)

            # Simulate forward environments
            curr_envs = self._clone_env(env, idx, num=self._samples) 
            for curr_env, action in zip(curr_envs, actions): self._simulate_env(curr_env, action)

            # Rollout trajectories
            traj_q_vals = self._parallel_rollout(curr_envs, self._branches)
            q_vals = getattr(np, self._mode)((q_vals, traj_q_vals), axis=0)

            # Update sampling distribution
            rank_order = q_vals.argsort()
            elites = actions[rank_order][:self._elites]
            self._update_distribution(elites)

            # Retain fraction of elites for next CEM-iteration
            decayed_elite_pop = (self._elites * self._keep_elites)
            num_elites = int(round(decayed_elite_pop * self._decay ** self._steps))
            init_actions = None if num_elites == 0 else elites[num_elites]

        return

    @abstractmethod
    def _parallel_rollout(self, curr_envs, branches):
        """Perform trajectory rollouts on task structure as defined by branches.
        Parallelize computation for faster trajectory simulation. Use this method 
        for model-based forward prediction when the state evolution can be batched.
        """
        raise NotImplementedError
