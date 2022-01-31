import numpy as np

from .base import Box2DBase


class Pick2D(Box2DBase):

    def __init__(self, **kwargs):
        """Pick2D gym environment.
        """
        super().__init__(**kwargs)
        self.setup_spaces()

    def setup_spaces(self):
        pass

    def reward(self):
        pass

    def reset(self):
        super().reset()
        observation = None
        return observation

    def step(self, action):
        # Apply forces to robot joints
        # Set block.fixedRotation = True if it touches ground.
        
        observation = None
        reward = 0.0
        done = False
        info = {}

        done = done or super().step()
        return observation, reward, done, info
