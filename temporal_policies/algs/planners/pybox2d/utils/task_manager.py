import os
import torch
from collections import defaultdict

from temporal_policies.utils.config import Config
import temporal_policies.envs.pybox2d as pybox2d_envs
from temporal_policies.utils.trainer import load_from_path


class TaskManager:

    def __init__(self, task, checkpoints, device="auto", load_models=True):
        """Handles loading and configuration of models and environments as per a task sequence.
        
        args:
            task: list of primitives (defined by gym env class names) and planning hyperparameters
                  to be executed in sequence            
            checkpoints: unordered list of checkpoints to trained primitives involved in the task
            load_models: whether or not to pre-load the model checkpoints
        """
        configs = [Config.load(os.path.join(os.path.dirname(c), "config.yaml")) for c in checkpoints]
        envs = [c["env"].split("-")[0] for c in configs]
        assert len(envs) == len(set(envs)), "Environment primitives must be unique"
        
        self._task = task
        self._configs = {env: v["env_kwargs"] for env, v in zip(envs, configs)}
        self._checkpoints = {env: v for env, v in zip(envs, checkpoints)}
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Preload models for efficiency
        self._load_models = load_models
        if self._load_models:
            self._models = {}
            for env in self._configs: self._models[env] = self._load_model(env)
    
        self._clean()
    
    def _load_model(self, key):
        """Return a loaded model.
        """
        if isinstance(key, int): key = self._task[key]["env"]
        model = load_from_path(self._get_checkpoint(key), device=self._device, strict=True)
        model.eval_mode()
        return model
    
    def _get_config(self, key):
        """Return environment configuration kwargs.
        """
        if isinstance(key, int): key = self._task[key]["env"]
        return self._configs[key]

    def _get_checkpoint(self, key):
        """Return model checkpoint.
        """
        if isinstance(key, int): key = self._task[key]["env"]
        return self._checkpoints[key]

    def _get_model(self, key):
        """Return a loaded or pre-loaded model.
        """
        if isinstance(key, int): key = self._task[key]["env"]
        model = self._models[key] if self._load_models else self._load_model(key)
        return model

    def _get_env_cls(self, key):
        """Return environment class.
        """
        if isinstance(key, int): key = self._task[key]["env"]
        return vars(pybox2d_envs)[key]
    
    def _clone_env(self, idx, env, num=None):
        """Return cloned environment(s).
        """
        if isinstance(env, list): return [self._clone_env(idx, e, num=num) for e in env]
        env_cls = self._get_env_cls(idx)
        config = self._get_config(idx)
        if num is None: return env_cls.clone(env, **config)
        return [env_cls.clone(env, **config) for _ in range(num)]

    def _load_env(self, idx, env, num=None):
        """Return loaded environment(s).
        """
        if isinstance(env, list): return [self._load_env(idx, e, num=num) for e in env]
        env_cls = self._get_env_cls(idx)
        config = self._get_config(idx)
        loaded_env = env_cls.load(env, **config)
        if num is None: return self._clone_env(idx, loaded_env)
        return self._clone_env(idx, loaded_env, num=num)

    def _trajectory_setup(self, opt, dep):
        """Construct branches for simulation and expected reward approximation, i.e, V(s), Q(s, a).
        args: 
            opt: list(int) of primitive indices to optimize under
            dep: nested list of integers (optimization variables) and dictionaries (simulation variables)
            
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
