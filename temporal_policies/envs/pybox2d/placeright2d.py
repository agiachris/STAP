from re import M
import numpy as np
from gym import spaces
from Box2D import *

from .base import Box2DBase
from .utils import shape_to_vertices


class PlaceRight2D(Box2DBase):

    def __init__(self, **kwargs):
        """PlaceRight2D gym environment.
        """
        super().__init__(**kwargs)
        self._setup_spaces()
        self.agent = None
        
    def reset(self):
        super().reset()
        self.agent = self._get_body("item", "block")

    def step(self, action):
        """Action components are activated via tanh().
        """
        # Act
        low, high = self.action_space.low, self.action_space.high
        action = low + (high - low) * ((action + 1) / 2)
        self.agent.position = b2Vec2(action[0], self.agent.position[1])
        self.agent.angle = action[1]
        self.agent.fixedRotation = True

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
        item_w = max(self._get_shape_kwargs("item")["size"])
        wksp_pos_x, wksp_pos_y = self._get_shape("playground", "ground")["position"]
        wksp_w, wksp_h = self._get_shape_kwargs("playground")["size"]
        wksp_t = self._get_shape_kwargs("playground")["t"]
        
        # Action space
        x_min = wksp_pos_x - wksp_w / 2 + item_w / 2
        x_max = wksp_pos_x + wksp_w / 2 - item_w / 2
        self.action_space = spaces.Box(
            low=np.array([x_min, -np.pi/2]),
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
        """PlaceRight2D reward function.
            - reward=1.0 iff block touches ground and to the right of receptacle box
            - reward=0.0 otherwise
        """
        on_ground = False
        for contact in self.agent.contacts:
            if contact.other.userData == self._get_body_name("playground", "ground"):
                on_ground = True
                break
        
        box_vertices = shape_to_vertices(
            position=self._get_body("box", "ceiling").position,
            box=self._get_shape("box", "ceiling")["box"]
        )
        x_min = np.amax(box_vertices, axis=0)[0]
        on_right = self.agent.position[0] >= x_min
        reward = float(on_ground and on_right)
        return reward

    def _get_done(self):
        return True
    