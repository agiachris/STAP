import numpy as np
from gym import spaces
from Box2D import *

from .base import Box2DBase
from .utils import plot

class PlaceRight2D(Box2DBase):

    def __init__(self, **kwargs):
        """PlaceRight2D gym environment.
        """
        super().__init__(**kwargs)
        self._setup_spaces()

    def step(self, action):
        """Action components are in [-1, 1].
        """
        # Act
        low = self.action_space.low
        high = self.action_space.high
        action = low + (high - low) * ((action + 1) / 2)
        item = self.env["item"]["bodies"]["block"]
        item.position = b2Vec2(action[0], item.position[1])
        item.angle = action[1]
        
        # Simulate
        steps_exceeded = super().step()
        observation = self._get_observation()
        reward = self._get_reward(observation)
        done = steps_exceeded or self._get_done()
        info = {}
        return observation, reward, done, info
    
    def _setup_spaces(self):
        """PlaceRight primitive action and observation spaces.

        Action space: x-dim coordinate and angle of "item"
        Observation space: bounding box parameters of all rigid bodies
        """
        item_w = max(self.env["item"]["shape_kwargs"]["size"])
        wksp_pos_x, wksp_pos_y = self.env["playground"]["shapes"]["ground"]["position"]
        wksp_w, wksp_h = self.env["playground"]["shape_kwargs"]["size"]
        wksp_t = self.env["playground"]["shape_kwargs"]["t"]
        
        # Action space
        x_min = wksp_pos_x - wksp_w / 2 + item_w / 2
        x_max = wksp_pos_x + wksp_w / 2 - item_w / 2
        self.action_space = spaces.Box(
            low=np.array([x_min, 0]),
            high=np.array([x_max, np.pi/2])
        )

        # Observation space
        x_min = wksp_pos_x - wksp_w / 2 - wksp_t
        x_max = wksp_pos_x + wksp_w / 2 + wksp_t
        y_min = wksp_pos_y - wksp_t / 2
        y_max = wksp_pos_y + wksp_t / 2 + wksp_h
        w_min, w_max = wksp_t / 2, wksp_w / 2 + wksp_t
        h_min, h_max = wksp_t / 2, wksp_h / 2
        n = len(self.world.bodies) * 4
        self.observation_space = spaces.Box(
            low=np.tile(np.array([x_min, y_min, w_min, h_min]), n), 
            high=np.tile(np.array([x_max, y_max, w_max, h_max]), n)
        )
        print("Action Space:\n", self.action_space)

    def _get_observation(self):
        k = 0
        observation = np.zeros((self.observation_space.shape[0]))
        for _, object_data in self.env.items():
            for shape_name, shape_data in object_data["shapes"].items():
                position = np.array(object_data["bodies"][shape_name].position)
                box = shape_data["box"] / 2
                observation[k: k+4] = np.concatenate((position, box))                
                k += 4
        return observation

    def _get_reward(self, observation):
        item = self.env["item"]["bodies"]["block"]

        # penalize collision w/ static object, reward anything else

        return 0

    def _get_done(self):
        return True
    