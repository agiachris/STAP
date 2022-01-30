from gym import Env
from abc import (ABC, abstractmethod)
from .generator import Generator
import matplotlib.pyplot as plt


class Box2DBase(Env, Generator):

    @abstractmethod
    def __init__(self, max_steps=1000, time_steps=1.0/60.0, vel_iters=10, pos_iters=10, **kwargs):
        """Box2D environment base class.
        """
        print("Init Box2DBase")
        super().__init__()

        self._time_steps = time_steps
        self._vel_iters = vel_iters
        self._pos_iters = pos_iters

        self.world = None
        self.agent = None
        self.state = None

    @abstractmethod
    def reset(self):
        """Reset environment state.
        """
        if self._world is not None:
            for body in self.world.bodies:
                self._world.DestroyBody(body) 
        self.__next__()

    @abstractmethod
    def step(self, action):
        """Take environment step at self._time_steps frequency.
        """
        self._world.Step(self._time_steps, self._vel_iters, self._pos_iters)
        self._world.ClearForces()

    def render(self):
        """Render sprites on all 2D bodies and set background to white.
        """
        raise NotImplementedError
