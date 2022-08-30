from typing import Any, Optional

import numpy as np
from gym import spaces

from .base import Box2DBase
from .utils import shape_to_vertices
from temporal_policies.controllers.siso_control import SISOControl
from temporal_policies.envs import base as envs


class PushLeft(envs.Primitive):
    def __init__(self):
        super().__init__(0, None)


class PushLeft2DControl(Box2DBase):
    def __init__(self, control_kwargs, **kwargs):
        """PushLeft2DControl gym environment."""
        super().__init__(**kwargs)
        self._base_kwargs = kwargs
        self._pid_control = SISOControl(**control_kwargs)

    def get_primitive(self) -> envs.Primitive:
        return self._primitive

    def set_primitive(
        self,
        primitive: Optional[envs.Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> envs.Env:
        return self

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> envs.Primitive:
        return self._primitive

    def step(self, action):
        """Action components are activated via tanh()."""
        # Act
        action = action.astype(float)
        low, high = self.action_scale.low, self.action_scale.high
        action = low + (high - low) * (action + 1) * 0.5
        self._pid_control.reset(
            ref=action.item(), y=self.agent.position[0], scale=float(self.agent.mass)
        )

        # Step once to let world settle.
        self.world.ClearForces()
        self.world.Step(self._time_steps, self._vel_iters, self._pos_iters)

        # Simulate
        terminated = False
        truncated = False
        reward = 0
        while not terminated and not truncated:
            u = self._pid_control.u(self.agent.position[0])
            self.agent.ApplyForce((u, 0), self.agent.position, wake=True)
            observation, rew, terminated, truncated, info = super().step()
            reward += rew
        return observation, reward, terminated, truncated, info

    def _setup_spaces(self):
        """PushLeft2DControl primitive action and observation spaces.
        Action space: (self.agent.position.x)
        Observation space: [Bounding box parameters of all 2D rigid bodies]
        """
        # Agent
        self.agent = self._get_body("item", "block")
        self.agent.fixedRotation = True

        # Space params
        item_w = max(self._get_shape_kwargs("item")["size"])
        wksp_pos_x, wksp_pos_y = self._get_shape("playground", "ground")["position"]
        wksp_w, wksp_h = self._get_shape_kwargs("playground")["size"]
        wksp_t = self._get_shape_kwargs("playground")["t"]

        # Action space
        x_min = wksp_pos_x - wksp_w * 0.5 + item_w * 0.5
        x_max = wksp_pos_x + wksp_w * 0.5 - item_w * 0.5
        self._primitive = PushLeft()
        self._primitive.action_scale = spaces.Box(
            low=np.array([x_min], dtype=np.float32),
            high=np.array([x_max], dtype=np.float32),
        )
        self._primitive.action_space = spaces.Box(
            low=np.array([-1], dtype=np.float32), high=np.array([1], dtype=np.float32)
        )

        # Observation space
        x_min = wksp_pos_x - wksp_w * 0.5 - wksp_t
        x_max = wksp_pos_x + wksp_w * 0.5 + wksp_t
        y_min = wksp_pos_y - wksp_t * 0.5
        y_max = wksp_pos_y + wksp_t * 0.5 + wksp_h
        w_min, w_max = wksp_t * 0.5, wksp_w * 0.5 + wksp_t
        h_min, h_max = wksp_t * 0.5, wksp_h * 0.5

        all_bodies = set([body.userData for body in self.world.bodies])
        redundant_bodies = set(
            [*self._get_bodies("playground").keys(), self.agent.userData]
        )
        self._observation_bodies = all_bodies - redundant_bodies
        reps = len(self._observation_bodies) + 1

        self.image_space = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        if self._image_observation:
            self.observation_space = self.image_space
        else:
            low = np.tile(np.array([x_min, y_min, w_min, h_min]), reps)
            low = np.concatenate((low, [-np.pi * 0.5 - 1e-2]))
            high = np.tile(np.array([x_max, y_max, w_max, h_max]), reps)
            high = np.concatenate((high, [np.pi * 0.5 + 1e-2]))
            self.observation_space = spaces.Box(
                low=low.astype(np.float32), high=high.astype(np.float32)
            )

    def _get_reward(self):
        """PushLeft2D reward function.
        - reward=1.0 iff block is pushed within the receptacle
        - reward=0.0 otherwise
        """
        reward = float(self.__in_box())
        return reward

    def __on_ground(self):
        for contact in self.agent.contacts:
            if contact.other.userData == self._get_body_name("playground", "ground"):
                return True
        return False

    def __on_right(self):
        box_vertices = shape_to_vertices(
            position=self._get_body("box", "ceiling").position,
            box=self._get_shape("box", "ceiling")["box"],
        )
        x_min = np.max(box_vertices, axis=0)[0]
        on_right = self.agent.position[0] >= x_min
        return on_right

    def __in_box(self):
        box_vertices = shape_to_vertices(
            position=self._get_body("box", "ceiling").position,
            box=self._get_shape("box", "ceiling")["box"],
        )
        y_max = np.max(box_vertices, axis=0)[1]

        touching_wall = False
        for contact in self.agent.contacts:
            if contact.other.userData == self._get_body_name("box", "wall"):
                touching_wall = True
                break

        in_box = touching_wall and self.agent.position[1] < y_max
        return in_box

    def __invalid_contact(self):
        valid_bodies = [
            self._get_body_name("playground", "ground"),
            self._get_body_name("box", "wall"),
        ]
        for contact in self.agent.contacts:
            if contact.other.userData not in valid_bodies:
                return True
        return False

    def _is_done(self):
        return self.__in_box()

    def _is_valid(self):
        return self.__on_ground() and not self.__invalid_contact()

    def _is_valid_start(self):
        self.simulate(
            time_steps=100,
            clear_forces=True,
            break_on_done=False,
            accrue_rewards=False,
        )
        return self.__on_ground() and self.__on_right()
