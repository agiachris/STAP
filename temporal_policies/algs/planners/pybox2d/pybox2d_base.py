import torch
from torch.distributions import MultivariateNormal
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

import temporal_policies.envs.pybox2d as pybox2d_envs
from temporal_policies.utils.trainer import load_from_path
from temporal_policies.utils import utils


class Box2DTrajOptim(ABC):

    def __init__(self, task, checkpoints, configs, device="auto", load_models=True):
        """Base class for trajectory optimization algorithms in the PyBox2D environment.
        
        args:
            task: list of primitives (gym environment class names) and planning hyperparameters 
                  defining a task when executed in sequence
            checkpoints: unordered list of paths of checkpoints to unique primitive models 
                         required for the task
            configs: list of environment configuration dictionaries corresponding to checkpoints
            device: device to run models on
            load_models: whether or not to pre-load the model checkpoints
        """
        envs = [c["env"].split("-")[0] for c in configs]
        assert len(envs) == len(set(envs)), "Environment primitives must be unique"
        
        self._task = task
        self._configs = {env: v["env_kwargs"] for env, v in zip(envs, configs)}
        self._checkpoints = {env: v for env, v in zip(envs, checkpoints)}

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._load_models = load_models
        if self._load_models:
            self._models = {}
            for env in self._configs: self._models[env] = self._load_model(env)
    
        self._clean()
    
    def _load_model(self, key):
        """Return a loaded model.
        """
        if isinstance(key, int):
            key = self._task[key]["env"]
        model = load_from_path(self._get_checkpoint(key), device=self._device, strict=True)
        model.eval_mode()
        return model
    
    def _get_config(self, key):
        """Return environment configuration kwargs.
        """
        if isinstance(key, int):
            key = self._task[key]["env"]
        return self._configs[key]

    def _get_checkpoint(self, key):
        """Return model checkpoint.
        """
        if isinstance(key, int):
            key = self._task[key]["env"]
        return self._checkpoints[key]

    def _get_model(self, key):
        """Return a loaded or pre-loaded model.
        """
        if isinstance(key, int):
            key = self._task[key]["env"]
        model = self._models[key] if self._load_models else self._load_model(key)
        return model

    def _get_env_cls(self, key):
        """Return environment class.
        """
        if isinstance(key, int):
            key = self._task[key]["env"]
        return vars(pybox2d_envs)[key]
    
    def _clone_env(self, env, idx, num=None):
        """Return cloned environment(s).
        """
        env_cls = self._get_env_cls(idx)
        config = self._get_config(idx)
        if num is None: return env_cls.clone(env, **config)
        return [env_cls.clone(env, **config) for _ in range(num)]

    def _load_env(self, env, idx, num=None):
        """Return loaded environment(s).
        """
        env_cls = self._get_env_cls(idx)
        config = self._get_config(idx)
        if num is None: return env_cls.load(env, **config)
        return [env_cls.load(env, **config) for _ in range(num)]

    @abstractmethod
    def plan(self, env, idx, mode="prod"):
        """Execute planning algorithm.

        args:
            env: Box2dBase gym environment
            idx: index in the task sequence
            mode: affordance aggregation mode - hasattr(np, mode) == True
        """
        self._clean()
        self._setup(env, idx, mode=mode)

    def _clean(self):
        """Clean clean parameters for next run.
        """
        self._env = None
        self._idx = None
        self._mode = None
        self._branches = None

    def _setup(self, env, idx, mode="prod"):
        """Setup attributes trajectory optimization.
        """
        self._env = env
        self._idx = idx
        self._mode = mode
        assert hasattr(np, mode), "Value aggregation mode not supported in Numpy"
        self._branches = self._trajectory_setup(self._task[idx]["opt"], self._task[idx]["dep"])

    def _trajectory_setup(self, opt, dep):
        """Construct branches for simulation and expected reward approximation, i.e, V(s), Q(s, a).
        args: 
            opt: list(int) of primitive indices to optimize under
            dep: list(list(int)) of primitive execution paths to simulate
            
        returns:
            branches: defaultdict(list) of optimization variables and simulation branches
        """
        assert (len(opt) == len(dep))
        branches = []
        visited = defaultdict(list)

        # Optimization variables
        for i in range(len(dep)):
            if not dep[i]:
                branches.append(opt[i])
                continue
            visited[dep[i].pop(0)].append(i)
        
        # Simulation branches
        for sim_idx in visited:
            rec_opt = [opt[i] for i in visited[sim_idx]]
            rec_dep = [dep[i] for i in visited[sim_idx]]
            branches.append({sim_idx: self._trajectory_setup(rec_opt, rec_dep)})
        
        return branches
    
    def _q_function(self, idx, states, actions):
        """Query the Q-function of model with states and actions.

        args: 
            idx: index in the task sequence
            states: np.array or torch.tensor of states
            actions: np.array or torch.tensor of actions
        """
        if states.ndim == 1 and actions.ndim == 2:
            states = np.tile(states, (actions.shape[0], 1))
        
        is_batched = states.ndim == 2 and states.ndim == 2
        if not is_batched: 
            states = utils.unsqueeze(states, 0)
            actions = utils.unsqueeze(actions, 0)

        model = self._get_model(idx)
        fmt = model._format_batch
        q1, q2 = model.network.critic(fmt(states), fmt(actions))
        q_vals = utils.to_np(torch.min(q1, q2))
        return q_vals

    def _policy(self, idx, states, samples=1, variance=None, **kwargs):
        """Query the policy for actions given states.
        """
        is_batched = True if states.ndim == 2 else False
        actions = self._get_model(idx).predict(states, is_batched=is_batched, **kwargs)
        if variance is None: 
            if samples > 1: actions = np.tile(actions, (samples, 1))
            return actions

        # Batch mean and covariance tensors
        if not is_batched: actions = utils.unsqueeze(actions, 0)
        cov = np.eye(actions.shape[1], dtype=np.float32)
        if isinstance(variance, float): cov *= variance
        elif isinstance(variance, list): cov = cov @ np.array(variance, dtype=np.float32)
        else: raise ValueError("variance must be float or list type")

        # Sample from Multivariate Gaussian
        loc = torch.from_numpy(actions).to(self._device)
        cov = torch.from_numpy(cov).to(self._device).tile(loc.size(0), 1, 1)
        dist = MultivariateNormal(loc=loc, covariance_matrix=cov)
        actions = torch.clamp(dist.sample((samples,)), -1, 1)

        if samples == 1: actions = actions.squeeze(0)
        else: actions = actions.transpose(1, 0)
        if not is_batched: actions = actions.squeeze(0)
        return utils.to_np(actions).astype(np.float32)

    @staticmethod
    def _simulate_env(env, action):
        """Simulate forward a single step environment (in-place) given an action.
        
        args:
            env: Box2DBase gym environment
            action: primitive action - np.array (env.action_space.shape[0],)
        """
        for _ in range(env._max_episode_steps):
            _, _, done, info = env.step(action)
            assert done, "Environment must be done in a single step"
            break
        return env, info["success"]
