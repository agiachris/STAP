import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy

from .utils.io_manager import IOManager


class Box2DPlannerBase(IOManager, ABC):
    def __init__(self, mode="prod", **kwargs):
        IOManager.__init__(self, **kwargs)
        self._mode = mode

    @abstractproperty
    def planner_settings(self):
        settings = {
            "default_actor": self._default_actor,
            "default_critic": self._default_critic,
            "default_dynamics": self._default_dynamics,
            "mode": self._mode,
        }
        return deepcopy(settings)

    @abstractmethod
    def plan(self, idx, env, mode="prod"):
        """Execute planning algorithm.

        args:
            idx: index in the task sequence
            env: Box2dBase gym environment
            mode: affordance aggregation mode - hasattr(np, mode) == True
        """
        self._clean()
        self._setup(idx, env, mode=mode)

    def _clean(self):
        """Clean clean parameters for next run."""
        self._env = None
        self._idx = None
        self._mode = None
        self._branches = None

    def _setup(self, idx, env, mode="prod"):
        """Setup attributes trajectory optimization."""
        self._env = env
        self._idx = idx
        self._mode = mode
        assert hasattr(np, mode), "Value aggregation mode not supported in Numpy"
        self._branches = self._trajectory_setup(
            self._task[idx]["opt"], self._task[idx]["dep"]
        )
