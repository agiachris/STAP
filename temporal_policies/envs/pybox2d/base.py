from gym import Env
from abc import (ABC, abstractmethod)
from skimage import draw
import numpy as np

from .generator import Generator
from .utils import (rigid_body_2d, shape_to_vertices)
from .constants import COLORS


class Box2DBase(ABC, Env, Generator):

    @abstractmethod
    def __init__(self, 
                 max_steps=1,
                 steps_per_action=1, 
                 time_steps=1.0/60.0, 
                 vel_iters=10, 
                 pos_iters=10, 
                 **kwargs):
        """Box2D environment base class.
        """
        super().__init__()
        self.max_steps = max_steps
        self._steps_per_action = steps_per_action
        self._time_steps = time_steps
        self._vel_iters = vel_iters
        self._pos_iters = pos_iters

    def reset(self):
        """Reset environment state.
        """
        if self.world is not None:
            for body in self.world.bodies:
                self.world.DestroyBody(body) 

        self.steps = 0
        self.__next__()
        self._setup_spaces()
        self._render_setup()
        observation = self._get_observation()
        return observation

    @abstractmethod
    def step(self, action=None):
        """Take environment steps at self._time_steps frequency.
        """
        for _ in range(self._steps_per_action):    
            self.world.Step(self._time_steps, self._vel_iters, self._pos_iters)
            self.world.ClearForces()
        
        self.steps += 1
        steps_exceeded = self.steps >= self.max_steps
        return steps_exceeded
    
    @abstractmethod
    def _setup_spaces(self):
        """Setup observation space, action space, and supporting attributes.
        """
        raise NotImplementedError
            
    @abstractmethod
    def _get_observation(self):
        """Observation model.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, observation):
        """Scalar reward function.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_done(self):
        """Returns True if terminal state has been reached.
        """

    def _render_setup(self, mode="human"):
        """Initialize rendering parameters.
        """
        if mode == "human":
            # Get image dimensions according the playground size
            workspace = self.env["playground"]["shape_kwargs"]
            (w, h), t = workspace["size"], workspace["t"]
            r = 1 / t
            assert r.is_integer()
            y_range = int((w + 2*t) * r)
            x_range = int((h + t) * r)
            image = np.ones((x_range + 1, y_range + 1, 3)) * 255
            
            # Resolve reference frame transforms
            global_to_workspace = rigid_body_2d(0, -self._t_global[0], -self._t_global[1], r)
            workspace_to_image = rigid_body_2d(np.pi/2, h, w/2 + t, r)
            global_to_image_px = workspace_to_image @ global_to_workspace

            # Render static world bodies
            static_image = image.copy()
            for _, object_data in self.env.items():
                if object_data["type"] != "static": continue
                for _, shape_data in object_data["shapes"].items():
                    vertices = shape_to_vertices(**shape_data) * r
                    vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
                    vertices_px = np.floor(global_to_image_px @ vertices.T).astype(int)
                    x_idx, y_idx = draw.polygon(vertices_px[0, :], vertices_px[1, :])

                    # Avoid rendering overlapping static objects
                    if object_data["class"] != "workspace":
                        static_mask = np.amin(static_image, axis=2) == 255
                        idx_filter = static_mask[x_idx, y_idx]
                        x_idx, y_idx = x_idx[idx_filter], y_idx[idx_filter]

                    color = object_data["render_kwargs"]["color"] if "render_kwargs" in object_data else "black"
                    static_image[x_idx, y_idx] = COLORS[color]

            # Rendering attributes
            self._r = r
            self._global_to_image_px = global_to_image_px
            self._image = image
            self._static_image = static_image
        
    def render(self, mode="human"):
        """Render sprites on all 2D bodies and set background to white.
        """
        if mode == "human":
        
            # Render all world shapes
            image = self._image.copy()
            for _, object_data in self.env.items():
                if object_data["type"] == "static": continue
                for shape_name, shape_data in object_data["shapes"].items():
                    position = np.array(object_data["bodies"][shape_name].position)
                    vertices = shape_to_vertices(np.zeros_like(position), shape_data["box"]).T
                    vertices = np.concatenate((vertices, np.ones((1, vertices.shape[1]))), axis=0)
                    # Must orient vertices by angle about object centroid
                    angle = object_data["bodies"][shape_name].angle
                    vertices = rigid_body_2d(angle, 0, 0) @ vertices
                    vertices[:2, :] = (vertices[:2, :] + np.expand_dims(position, 1)) * self._r

                    vertices_px = np.floor(self._global_to_image_px @ vertices).astype(int)
                    x_idx, y_idx = draw.polygon(vertices_px[0, :], vertices_px[1, :], image.shape)
                    color = object_data["render_kwargs"]["color"] if "render_kwargs" in object_data else "navy"
                    image[x_idx, y_idx] = COLORS[color]
            
            x_idx, y_idx = np.where(np.amin(image, axis=2) < 255)
            static_image = self._static_image.copy()
            static_image[x_idx, y_idx, :] = image[x_idx, y_idx, :]
            image = static_image / 255
            return image.copy()
