import numpy as np
from .base import Box2DBase


class Pick2D(Box2DBase):

    def __init__(self, **kwargs):
        """Pick2D gym environment.
        """
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        # combine image and full state
        observation = None
        return observation

    def step(self, action):
        # Apply forces to robot joints
        # Set block.fixedRotation = True if it touches ground.
        super().step(action)
        observation = None
        reward = 0.0
        done = False
        info = {}
        return observation, reward, done, info
