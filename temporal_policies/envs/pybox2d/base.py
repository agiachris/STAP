from collections import defaultdict
from copy import deepcopy
import pathlib
from typing import Any, Optional, Set, Union

import Box2D
import gym
import numpy as np
from PIL import Image
from skimage import draw
import torch

from .generator import Generator
from .utils import rigid_body_2d, shape_to_vertices, to_homogenous
from .visualization import draw_caption
from temporal_policies import agents
from temporal_policies.envs import base
from temporal_policies.utils import recording, tensors


class Box2DBase(base.Env, Generator):
    metadata = {"render_modes": ["human", "rgb_array", "default"]}

    def __init__(
        self,
        name: str,
        max_episode_steps: int,
        steps_per_action: int,
        image_observation: bool = False,
        observation_noise: float = 0.0,
        time_steps: float = 1.0 / 60.0,
        vel_iters: int = 10,
        pos_iters: int = 10,
        clear_forces: bool = True,
        break_on_done: bool = True,
        cumulative_reward: float = 0.0,
        steps: int = 0,
        physics_steps: int = 0,
        physics_steps_buffer: int = 0,
        recording_freq: int = 3,
        **kwargs,
    ):
        """Box2D environment base class.

        args:
            name: Environment name
            max_episode_steps: maximum number per episode
            steps_per_action: number of simulation steps per action
            image_observation: Whether to use image or low-dimensional state observations
            observation_noise: percent noise added to observations
            time_steps: simulation frequency
            vel_iters: Box2D velocity numerical solver iterations per time step
            pos_iters: Box2D positional numerical solver iterations per time step
            clear_forces: clear forces upon every Box2D simulation step
            break_on_done: stop simulation at terminal state
            cumulative_reward: rewards accrued over the course of the episode
            steps: initial step of the gym environment
            physics_steps: number of time_steps occured
            physics_steps_buffer: number of extra time_steps after episode has terminated
            recording_freq: Recording frequency
        """
        Generator.__init__(self, **kwargs)
        self.name = name
        self._max_episode_steps = max_episode_steps
        self._steps_per_action = steps_per_action
        self._image_observation = image_observation
        self._observation_noise = observation_noise
        self._time_steps = time_steps
        self._vel_iters = vel_iters
        self._pos_iters = pos_iters
        self._clear_forces = clear_forces
        self._break_on_done = break_on_done
        self._cumulative_reward = cumulative_reward
        self._steps = steps
        self._physics_steps = physics_steps
        self._physics_steps_buffer = physics_steps_buffer

        self._recorder = recording.Recorder(frequency=recording_freq)

        self._setup_spaces()
        self._render_setup()
        self._eval_mode = False
        self.render_mode = "human"

        # Construct state space.
        assert self.env is not None
        self._num_bodies = sum(
            len(self._get_shapes(object_name)) for object_name in self.env
        )
        ground = self._get_shape("playground", "ground")
        x, y = ground["position"]
        workspace = self._get_shape_kwargs("playground")
        w, h = workspace["size"]
        low = np.array([x - 0.5 * w, y, -np.pi / 2, -1e3, -1e3, -1e3], dtype=np.float32)
        high = np.array(
            [x + 0.5 * w, y + h, np.pi / 2, 1e3, 1e3, 1e3], dtype=np.float32
        )
        self.state_space = gym.spaces.Box(
            low=np.tile(low, self.num_bodies),
            high=np.tile(high, self.num_bodies),
        )

        self.agent: Optional[Box2D.b2Body] = None
        self._observation_bodies: Set[Box2D.b2Body] = set()

    def _clean_base_kwargs(self):
        """Clean up base kwargs for future envioronment loading and cloning."""
        assert hasattr(self, "_base_kwargs")
        if "env" in self._base_kwargs:
            del self._base_kwargs["env"]
        if "world" in self._base_kwargs:
            del self._base_kwargs["world"]

    def train_mode(self):
        """Set environment to train mode."""
        self._eval_mode = False
        self._break_on_done = True
        self._physics_steps_buffer = 0

    def eval_mode(self, break_on_done=True, physics_steps_buffer=10):
        """Set environment to evaluation mode."""
        self._eval_mode = True
        self._break_on_done = break_on_done
        self._physics_steps_buffer = physics_steps_buffer

    @classmethod
    def load(cls, env, **kwargs):
        """Load environment from pre-existing b2World. The b2World instance is transferred
        into the new environment, and hence modifying (simulating) the previous environment
        will modify the new environment, and vice-versa.

        args:
            env: unwrapped gym environment of Box2DBase subclass
            **kwargs: keyword arguments of Box2DBase subclass
        """
        env_kwargs = {
            # Box2DBase kwargs
            "cumulative_reward": env._cumulative_reward,
            "physics_steps": env._physics_steps,
            # Generator kwargs
            "env": env.env,
            "world": env.world,
            "mode": "load",
            # GeometryHandler kwargs
            "global_x": env._t_global[0],
            "global_y": env._t_global[1],
        }
        env_kwargs.update(deepcopy(kwargs))
        loaded_env = cls(**env_kwargs)
        loaded_env._clean_base_kwargs()

        # Use same recorder instance for rendering continuity.
        loaded_env._recorder = env._recorder

        if env._eval_mode:
            loaded_env.eval_mode(
                break_on_done=env._break_on_done,
                physics_steps_buffer=env._physics_steps_buffer,
            )
        return loaded_env

    @classmethod
    def clone(cls, env, **kwargs):
        """Clone pre-existing b2World environment. A b2World instance is created
        to directly replicate the state of the current environment.

        args:
            env: unwrapped gym environment of Box2DBase subclass
            **kwargs: keyword arguments of Box2DBase subclass
        """
        env_kwargs = {
            # Box2DBase kwargs
            "cumulative_reward": env._cumulative_reward,
            "steps": env._steps,
            "physics_steps": env._physics_steps,
            # Generator kwargs
            "env": env.env,
            "world": env.world,
            "mode": "clone",
            # GeometryHandler kwargs
            "global_x": env._t_global[0],
            "global_y": env._t_global[1],
        }
        env_kwargs.update(deepcopy(kwargs))
        loaded_env = cls(**env_kwargs)
        loaded_env._clean_base_kwargs()
        loaded_env._recorder = deepcopy(loaded_env.recorder)
        if env._eval_mode:
            loaded_env.eval_mode(
                break_on_done=env._break_on_done,
                physics_steps_buffer=env._physics_steps_buffer,
            )
        return loaded_env

    def reset(self):
        """Reset environment state."""
        if self.world is not None:
            for body in self.world.bodies:
                self.world.DestroyBody(body)

        is_valid_start = False
        while not is_valid_start:
            next(self)
            self._setup_spaces()
            is_valid_start = self._is_valid_start()
            self._cumulative_reward = 0.0
            self._steps = 0
            self._physics_steps = 0
        self._render_setup()

        observation = self.get_observation()
        return observation, {}

    def step(self, action=None):
        """Take environment steps at self._time_steps frequency."""
        obs, reward, terminated, truncated, info = self.simulate()
        self._steps += 1
        if self._steps >= self._max_episode_steps:
            terminated = True
            truncated = False
            info["success"] = info.get("success", False)
        return obs, reward, terminated, truncated, info

    def simulate(
        self,
        time_steps=None,
        clear_forces=None,
        break_on_done=None,
        accrue_rewards=True,
    ):
        # Custom simulation arguments
        time_steps = time_steps if time_steps is not None else self._steps_per_action
        clear_forces = clear_forces if clear_forces is not None else self._clear_forces
        break_on_done = (
            break_on_done if break_on_done is not None else self._break_on_done
        )

        obs = None
        done = False
        reward = 0
        info = {}
        steps = 0
        while steps < time_steps:
            self.world.Step(self._time_steps, self._vel_iters, self._pos_iters)
            if clear_forces:
                self.world.ClearForces()

            # Only accrue rewards for valid states
            is_valid = self._is_valid()
            if accrue_rewards and not done:
                r = 0 if not is_valid else self._get_reward()
                reward += r
                self._cumulative_reward += r

            self._recorder.add_frame(self.render)
            self._physics_steps += 1
            steps += 1

            if not done:
                # Save terminal state
                is_done = self._is_done()
                if is_done or not is_valid:
                    done = True
                    obs = self.get_observation()
                    info["success"] = is_done and is_valid and reward > 0

                    # Optionally run extra simulation steps
                    if break_on_done:
                        steps = 0
                        time_steps = self._physics_steps_buffer
                        clear_forces = True

        if not done:
            obs = self.get_observation()

        self._recorder.add_frame(self.render, override_frequency=True)

        self.world.ClearForces()
        truncated = not is_valid
        terminated = done and not truncated
        return obs, reward, terminated, truncated, info

    @property
    def num_bodies(self) -> int:
        """Number of environment bodies."""
        return self._num_bodies

    def get_state(self) -> np.ndarray:
        """Gets the environment state.

        [N * 3] array of mutable body properties (position, angle).
        """
        state = []
        assert self.env is not None

        for object_name in self.env:
            for shape_name, shape_data in self._get_shapes(object_name).items():
                body = self._get_body(object_name, shape_name)

                body_state = np.zeros(6, dtype=np.float32)
                body_state[:2] = body.position
                body_state[2] = body.angle
                body_state[3:5] = body.linearVelocity
                body_state[5] = body.angularVelocity

                state.append(body_state)

        return np.concatenate(state, axis=0)

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        assert self.env is not None
        iter_state = iter(np.reshape(state, (-1, 6)))
        for object_name in self.env:
            for shape_name in self._get_shapes(object_name):
                body = self._get_body(object_name, shape_name)
                body_state = next(iter_state).astype(float)

                body.position = Box2D.b2Vec2(*body_state[:2])
                body.angle = body_state[2]
                body.linearVelocity = body_state[3:5]
                body.angularVelocity = body_state[5]
                body.awake = True
        assert self.world is not None
        self.world.ClearForces()

        self._cumulative_reward = 0.0
        self._steps = 0
        self._physics_steps = 0

        return True

    def _setup_spaces(self):
        """Setup observation space, action space, and supporting attributes."""
        raise NotImplementedError

    def get_observation(self, image: Optional[bool] = None):
        """Observation model. Optionally incorporate noise to observations."""
        if image is None:
            image = self._image_observation

        if image:
            old_render_mode = self.render_mode
            self.render_mode = "rgb_array"
            img = self.render()
            self.render_mode = old_render_mode
            return img

        k = 0
        observation = np.zeros((self.observation_space.shape[0]), dtype=np.float32)
        assert self.env is not None
        for object_name in self.env:
            for shape_name, shape_data in self._get_shapes(object_name).items():
                if shape_name not in self._observation_bodies:
                    continue
                position = np.array(
                    self._get_body(object_name, shape_name).position, dtype=np.float32
                )
                observation[k : k + 4] = np.concatenate((position, shape_data["box"]))
                k += 4
        # Agent data
        assert self.agent is not None
        position = np.array(self.agent.position, dtype=np.float32)
        box = self._get_shape("item", "block")["box"]
        angle = np.array([self.agent.angle])
        observation[k : k + 5] = np.concatenate((position, box, angle))

        # Add noise.
        low = self.observation_space.low
        high = self.observation_space.high
        margin = (high - low) * self._observation_noise
        noise = np.random.uniform(-margin, margin)
        observation = np.clip(observation + noise, low, high)
        return observation.astype(np.float32)

    def _get_reward(self):
        """Scalar reward function."""
        raise NotImplementedError

    def _is_done(self):
        """Returns True if terminal state has been reached."""
        raise NotImplementedError

    def _is_valid_start(self):
        """Check if start state is valid."""
        raise NotImplementedError

    def _is_valid(self):
        """Check if current state is valid."""
        raise NotImplementedError

    def action_value(self, model, step=False):
        """Compute Q(s, a) at the current state and with policy predicted action.

        args:
            model: algs.Base.Algorithm instance containing trained models
        returns:
            q: Q(s, a) evaluate at the current state and policy predicted action
            output: state, action, cloned environment simulated forward under (s, a)
        """
        assert issubclass(
            type(self), Box2DBase
        ), "Must be called from a subclass of envs.pybox2d.base.Box2DBase"
        assert isinstance(
            model, agents.RLAgent
        ), "Model argument must be an instance of algs.RLAgent"

        # Tensorize state and action
        fmt = model._format_batch
        obs = self.get_observation()
        state = fmt(tensors.unsqueeze(obs, 0))
        action = model.predict(state, is_batched=True)

        assert state.size(0) == action.size(0)
        assert state.size(1) == self.observation_space.shape[0]
        assert action.size(1) == self.action_space.shape[0]

        # Infer action values
        q1, q2 = model.network.critic(state, action)
        q = tensors.numpy(torch.min(q1, q2)).item()

        output = {
            "state": tensors.numpy(state).squeeze(0),
            "action": tensors.numpy(action).squeeze(0),
        }
        # Simulate forward cloned environment
        if step:
            env = type(self).clone(self, **self._base_kwargs)
            output["env"] = env
            obs, rew, terminated, truncated, info = env.step(output["action"])
            output["observation"] = obs
            output["reward"] = rew
            output["done"] = terminated or truncated
            output["info"] = info

        return q, output

    def value_over_interp_actions(self, model, num, dims, step=True):
        """Compute Q(s, a) over linearly interpolated samples across the action space.

        args:
            model: algs.Base.Algorithm instance containing trained models
            num: number of evenly spaced action space samples
            dims: tuple of dimensions to interpolate across
        returns:
            qs: Q(s, a) evaluated across action space samples -- np.array (num)
            outputs: interpolated states, actions, action_dims, cloned environments simulated forward under (s, a) pairs
        """
        assert issubclass(
            type(self), Box2DBase
        ), "Must be called from a subclass of envs.pybox2d.base.Box2DBase"
        assert isinstance(
            model, agents.RLAgent
        ), "Model argument must be an instance of algs.Base.Algorithm"
        assert all(d in list(range(self.action_space.shape[0])) for d in dims)

        # Tensorize states and actions
        fmt = model._format_batch
        obs = self.get_observation()
        states = fmt(np.tile(tensors.unsqueeze(obs, 0), (num, 1)))
        default = model.predict(obs)
        actions, action_dims = self._interp_actions(num, dims, default=default)
        actions = fmt(actions)
        assert states.size(0) == actions.size(0)
        assert states.size(1) == self.observation_space.shape[0]
        assert actions.size(1) == self.action_space.shape[0]

        # Infer action values
        q1s, q2s = model.network.critic(states, actions)
        qs = tensors.numpy(torch.min(q1s, q2s))

        outputs = defaultdict(list)
        for state, action, action_dim in zip(
            tensors.numpy(states), tensors.numpy(actions), action_dims
        ):
            outputs["state"].append(state)
            outputs["action"].append(action)
            outputs["action_dim"].append(action_dim)
            if not step:
                continue
            # Simulate forward cloned environments
            env = type(self).clone(self, **self._base_kwargs)
            outputs["env"].append(env)
            obs, rew, terminated, truncated, info = env.step(action)
            outputs["observation"].append(obs)
            outputs["reward"].append(rew)
            outputs["done"].append(terminated or truncated)
            outputs["info"].append(info)

        return qs, outputs

    def value_over_interp_states(self, model, num, dims, step=False):
        """Compute Q(s, a) over linearly interpolated samples across the state space.

        args:
            model: algs.Base.Algorithm instance containing trained models
            num: number of evenly spaced state space samples
            dims: tuple of dimensions to interpolate across
        returns:
            qs: Q(s, a) evaluated across state space samples
            outputs: interpolated states, state_dims, actions, cloned environments simulated forward under (s, a) pairs
        """
        assert issubclass(
            type(self), Box2DBase
        ), "Must be called from a subclass of envs.pybox2d.base.Box2DBase"
        assert isinstance(
            model, agents.RLAgent
        ), "Model argument must be an instance of agents.RLAgent"
        assert all(d in list(range(self.observation_space.shape[0])) for d in dims)

        # Tensorize states and actions
        fmt = model._format_batch
        default = self.get_observation()
        states, state_dims = self._interp_states(num, dims, default=default)
        states = fmt(states)
        actions = model.predict(states, is_batched=True)
        assert states.size(0) == actions.size(0)
        assert states.size(1) == self.observation_space.shape[0]
        assert actions.size(1) == self.action_space.shape[0]

        # Infer action values
        q1s, q2s = model.network.critic(states, actions)
        qs = tensors.numpy(torch.min(q1s, q2s))

        outputs = defaultdict(list)
        for state, state_dim, action in zip(states, state_dims, actions):
            outputs["state"].append(state)
            outputs["state_dim"].append(state_dim)
            outputs["action"].append(action)
            if not step:
                continue
            # Simulate forward cloned environments
            env = type(self).clone(self, **self._base_kwargs)
            outputs["env"].append(env)
            obs, rew, terminated, truncated, info = env.step(action)
            outputs["observation"].append(obs)
            outputs["reward"].append(rew)
            outputs["done"].append(terminated or truncated)
            outputs["info"].append(info)

        return qs, outputs

    def _interp_actions(self, num, dims, default=None):
        """Linear interpolation of action space across specified dimensions.

        args:
            num: number of evenly spaced action space samples
            dims: tuple of dimensions to interpolate across
            default: default action values for unspecified dimensions -- np.array (action_space.ndim,)
        returns:
            actions: linear interpolation of action space -- np.array (num, action_space.ndim)
            action_dims: action components indexed at dims
        """
        mask = np.zeros(self.action_space.shape[0], dtype=bool)
        mask[dims] = True

        # Compute combinations of action components across specified dims
        low, high = self.action_space.low, self.action_space.high
        action_dims = np.linspace(low[mask], high[mask], num, dtype=np.float32)
        action_dims = np.array(np.meshgrid(*action_dims.T))
        action_dims = action_dims.T.reshape(-1, self.action_space.shape[0])

        if default is None:
            default = (low + high) * 0.5
        actions = np.tile(default, (action_dims.shape[0], 1))
        actions[:, mask] = action_dims.copy()

        assert all(self.action_space.contains(a) for a in actions)
        # Convert to un-normalized action space
        low, high = self.action_scale.low, self.action_scale.high
        action_dims = (low + (high - low) * (actions + 1) * 0.5)[:, dims].copy()
        return actions, action_dims

    def _interp_states(self, num, dims, default=None):
        """Linear interpolation of state space across specified dimensions.

        args:
            num: number of evenly spaced state space samples
            dims: tuple of dimensions to interpolate across
            default: default state values for unspecified dimensions -- np.array (observation_space.ndim,)
        returns:
            states: linear interpolation of state space -- np.array (num, observation_space.ndim)
            state_dims: state components indexed by dims
        """
        mask = np.zeros(self.observation_space.shape[0], dtype=bool)
        mask[dims] = True

        # Compute combinations of action components across specified dims
        low, high = self.observation_space.low, self.observation_space.high
        state_dims = np.linspace(low[mask], high[mask], num, dtype=np.float32)
        state_dims = np.array(np.meshgrid(*state_dims.T))
        state_dims = state_dims.T.reshape(-1, self.observation_space.shape[0])

        if default is None:
            default = (low + high) * 0.5
        states = np.tile(default, (state_dims.shape[0], 1))
        states[:, mask] = state_dims.copy()

        # Convert to un-normalized state space
        assert all(self.observation_space.contains(s) for s in states)
        state_dims = states[:, mask].copy()
        return states, state_dims

    def _render_setup(self, mode="human"):
        """Initialize rendering parameters."""
        if mode == "human" or mode == "rgb_array":
            # Get image dimensions according the playground size
            workspace = self._get_shape_kwargs("playground")
            (w, h), t = workspace["size"], workspace["t"]
            r = 1 / t
            assert r.is_integer()
            y_range = int((w + 2 * t) * r)
            x_range = int((h + t) * r)
            image = np.ones((x_range + 1, y_range + 1, 3), dtype=np.float32) * 255

            # Resolve reference frame transforms
            global_to_workspace = rigid_body_2d(
                0, -self._t_global[0], -self._t_global[1], r
            )
            workspace_to_image = rigid_body_2d(np.pi * 0.5, h, w * 0.5 + t, r)
            global_to_image = workspace_to_image @ global_to_workspace

            # Render static world bodies
            static_image = image.copy()
            for object_name in self.env.keys():
                if self._get_type(object_name) != "static":
                    continue
                for _, shape_data in self._get_shapes(object_name).items():
                    vertices = to_homogenous(shape_to_vertices(**shape_data))
                    vertices[:, :2] *= r
                    vertices_px = np.floor(global_to_image @ vertices.T).astype(int)
                    x_idx, y_idx = draw.polygon(vertices_px[0, :], vertices_px[1, :])

                    # Avoid rendering overlapping static objects
                    if self._get_class(object_name) != "workspace":
                        static_mask = np.min(static_image, axis=2) == 255
                        idx_filter = static_mask[x_idx, y_idx]
                        x_idx, y_idx = x_idx[idx_filter], y_idx[idx_filter]

                    static_image[x_idx, y_idx] = self._get_color(object_name)

            # Rendering attributes
            self._r = r
            self._global_to_image = global_to_image
            image_to_plot = rigid_body_2d(-np.pi * 0.5, 0, image.shape[0])
            self._global_to_plot = image_to_plot @ global_to_image
            self._image = image
            self._static_image = static_image

    def render(self):
        """Render sprites on all 2D bodies and set background to white."""
        mode = self.render_mode
        # Render all world shapes
        dynamic_image = self._image.copy()
        for object_name in self.env.keys():
            if self._get_type(object_name) == "static":
                continue
            for shape_name, shape_data in self._get_shapes(object_name).items():
                body = self._get_body(object_name, shape_name)
                position = np.array(body.position, dtype=np.float32)
                vertices = to_homogenous(
                    shape_to_vertices(np.zeros_like(position), shape_data["box"])
                ).T

                # Must orient vertices by angle about object centroid
                vertices = rigid_body_2d(body.angle, 0, 0) @ vertices
                vertices[:2, :] = (
                    vertices[:2, :] + np.expand_dims(position, 1)
                ) * self._r

                vertices_px = np.floor(self._global_to_image @ vertices).astype(int)
                x_idx, y_idx = draw.polygon(
                    vertices_px[0, :], vertices_px[1, :], dynamic_image.shape
                )
                dynamic_image[x_idx, y_idx] = self._get_color(object_name)

        x_idx, y_idx = np.where(np.min(dynamic_image, axis=2) < 255)
        image = self._static_image.copy()
        image[x_idx, y_idx, :] = dynamic_image[x_idx, y_idx, :]
        image = np.round(image).astype(np.uint8)

        if mode == "human":
            width, height = 480, 360
            caption = self._render_caption()
            dtype = np.uint8
        elif mode == "rgb_array":
            width, height = 64, 64
            caption = None
            dtype = np.uint8
        elif mode == "default":
            width, height = None, None
            caption = None
            dtype = np.uint8
        else:
            raise NotImplementedError(f"Rendering for mode {mode} is not suppported.")

        image = self._render_util(
            image, caption=caption, width=width, height=height, dtype=dtype
        )
        return image

    def _render_util(self, image, caption=None, width=480, height=360, dtype=np.uint8):
        """Caption and resize image."""
        image = Image.fromarray(image, "RGB")
        if width is not None or height is not None:
            image = image.resize((width, height))
        if caption:
            image = draw_caption(np.asarray(image), caption)
        return np.asarray(image, dtype=dtype)

    def _render_caption(self):
        """Return standard caption for the image."""
        caption = f"Env: {type(self).__name__} | Step: {self._steps} | "
        caption += f"Time: {self._physics_steps} | Reward: {self._cumulative_reward}"
        return caption

    def record_start(
        self,
        prepend_id: Optional[Any] = None,
        frequency: Optional[int] = None,
        mode: str = "default",
    ) -> bool:
        """Starts recording the simulation.

        Args:
            prepend_id: Prepends the new recording with the existing recording
                saved under this id.
            frequency: Recording frequency.
            mode: Recording mode. Options:
                - 'default': record at fixed frequency.
        Returns:
            Whether recording was started.
        """
        if isinstance(prepend_id, np.ndarray):
            prepend_id = prepend_id.item()
        if prepend_id is not None:
            prepend_id = str(prepend_id)

        if mode == "default":
            self._recorder.start(prepend_id, frequency)
        else:
            return False

        return True

    def record_stop(self, save_id: Optional[Any] = None, mode: str = "default") -> bool:
        """Stops recording the simulation.

        Args:
            save_id: Saves the recording to this id.
            mode: Recording mode. Options:
                - 'default': record at fixed frequency.
        Returns:
            Whether recording was stopped.
        """
        if isinstance(save_id, np.ndarray):
            save_id = save_id.item()
        if save_id is not None:
            save_id = str(save_id)

        if mode == "default":
            return self._recorder.stop(save_id)
        else:
            return False

    def record_save(
        self,
        path: Union[str, pathlib.Path],
        reset: bool = True,
        mode: Optional[str] = None,
    ) -> bool:
        """Saves all the recordings.

        Args:
            path: Path for the recording.
            reset: Reset the recording after saving.
            mode: Recording mode to save. If None, saves all recording modes.
        Returns:
            Whether any recordings were saved.
        """
        is_saved = False
        if mode is None or mode == "default":
            is_saved |= self._recorder.save(path, reset)

        return is_saved
