import torch
import numpy as np
from collections import defaultdict
from copy import deepcopy
from gym import Env
from abc import ABC, abstractmethod
from skimage import draw
from PIL import Image

from .generator import Generator
from .utils import rigid_body_2d, shape_to_vertices, to_homogenous
from .visualization import draw_caption
from temporal_policies.algs.base import Algorithm
import temporal_policies.utils.utils as utils


class Box2DBase(ABC, Env, Generator):

    @abstractmethod
    def __init__(self, 
                 max_episode_steps,
                 steps_per_action, 
                 time_steps=1.0/60.0, 
                 vel_iters=10, 
                 pos_iters=10,
                 cumulative_reward=0.0,
                 steps=0,
                 physics_steps=0,
                 **kwargs):
        """Box2D environment base class.

        args:
            max_episode_steps: maximum number per episode
            steps_per_action: number of simulation steps per action
            time_steps: simulation frequency
            vel_iters: Box2D velocity numerical solver iterations per time step  
            pos_iters: Box2D positional numerical solver iterations per time step  
            cumulative_reward: rewards accrued over the course of the episode
        """
        Generator.__init__(self, **kwargs)
        self._max_episode_steps = max_episode_steps
        self._steps_per_action = steps_per_action
        self._time_steps = time_steps
        self._vel_iters = vel_iters
        self._pos_iters = pos_iters
        self._cumulative_reward = cumulative_reward
        self.steps = steps
        self.physics_steps = physics_steps
        self._setup_spaces()
        self._render_setup()

    def _clean_base_kwargs(self):
        """Clean up base kwargs for future envioronment loading and cloning.
        """
        assert hasattr(self, "_base_kwargs")
        if "env" in self._base_kwargs: del self._base_kwargs["env"]
        if "world" in self._base_kwargs: del self._base_kwargs["world"]
    
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
            "env": env.env,
            "world": env.world,
            "geometry_params": {
                "global_x": env._t_global[0],
                "global_y": env._t_global[1]
            },
            "cumulative_reward": env._cumulative_reward,
            "physics_steps": env.physics_steps,
            "mode": "load"
        }
        env_kwargs.update(deepcopy(kwargs))
        env = cls(**env_kwargs)
        env._clean_base_kwargs()
        return env

    @classmethod
    def clone(cls, env, **kwargs):
        """Clone pre-existing b2World environment. A b2World instance is created 
        to directly replicate the state of the current environment. 

        args:
            env: unwrapped gym environment of Box2DBase subclass
            **kwargs: keyword arguments of Box2DBase subclass
        """
        env_kwargs = {
            "env": env.env,
            "world": env.world,
            "geometry_params": {
                "global_x": env._t_global[0],
                "global_y": env._t_global[1]
            },
            "cumulative_reward": env._cumulative_reward,
            "steps": env.steps,
            "physics_steps": env.physics_steps,
            "mode": "clone"
        }
        env_kwargs.update(deepcopy(kwargs))
        env = cls(**env_kwargs)
        env._clean_base_kwargs()
        return env

    @abstractmethod
    def reset(self):
        """Reset environment state.
        """
        if self.world is not None:
            for body in self.world.bodies:
                self.world.DestroyBody(body) 

        self._cumulative_reward = 0.0
        self.steps = 0
        self.physics_steps = 0
        
        is_valid = False
        while not is_valid:
            next(self)
            self._setup_spaces()
            is_valid = self._is_valid()
        self._render_setup()
        
        observation = self._get_observation()
        return observation

    @abstractmethod
    def step(self, clear_forces=True, render=False):
        """Take environment steps at self._time_steps frequency.
        """
        self.simulate(self._steps_per_action, clear_forces, render)
        if render: self._render_buffer.append(self.render())
        self.steps += 1
        steps_exceeded = self.steps >= self._max_episode_steps
        return steps_exceeded
    
    def simulate(self, time_steps, clear_forces=True, render=False):
        done_reward = False
        for _ in range(time_steps):
            self.world.Step(self._time_steps, self._vel_iters, self._pos_iters)
            if clear_forces: self.world.ClearForces()
            if not self._is_done(): 
                self._cumulative_reward += self._get_reward()
            elif self._is_done() and not done_reward:
                self._cumulative_reward += self._get_reward()
                done_reward = True
            if render: self._render_buffer.append(self.render())            
            self.physics_steps += 1
        self.world.ClearForces()

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
    def _get_reward(self):
        """Scalar reward function.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _is_done(self):
        """Returns True if terminal state has been reached.
        """
        raise NotImplementedError

    @abstractmethod
    def _is_valid(self):
        """Check if start state is valid.
        """
        raise NotImplementedError

    def action_value(self, model, step=False):
        """Compute Q(s, a) at the current state and with policy predicted action.

        args:
            model: algs.Base.Algorithm instance containing trained models
        returns:
            q: Q(s, a) evaluate at the current state and policy predicted action
            output: state, action, cloned environment simulated forward under (s, a)
        """
        assert issubclass(type(self), Box2DBase), "Must be called from a subclass of envs.pybox2d.base.Box2DBase"
        assert isinstance(model, Algorithm), "Model argument must be an instance of algs.Base.Algorithm"
        
        # Tensorize state and action
        fmt = model._format_batch
        obs = self._get_observation()
        state = fmt(utils.unsqueeze(obs, 0))
        action = model.predict(state, is_batched=True)

        assert state.size(0) == action.size(0)
        assert state.size(1) == self.observation_space.shape[0]
        assert action.size(1) == self.action_space.shape[0]

        # Infer action values
        q1, q2 = model.network.critic(state, action)
        q = utils.to_np(torch.min(q1, q2)).item()

        output = {
            "state": utils.to_np(state).squeeze(0),
            "action": utils.to_np(action).squeeze(0)
        }
        # Simulate forward cloned environment
        if step:
            env = type(self).clone(self, **self._base_kwargs)
            output["env"] = env
            obs, rew, done, info = env.step(output["action"])
            output["observation"] = obs
            output["reward"] = rew
            output["done"] = done
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
        assert issubclass(type(self), Box2DBase), "Must be called from a subclass of envs.pybox2d.base.Box2DBase"
        assert isinstance(model, Algorithm), "Model argument must be an instance of algs.Base.Algorithm"
        assert all(d in list(range(self.action_space.shape[0])) for d in dims)

        # Tensorize states and actions
        fmt = model._format_batch
        obs = self._get_observation()
        states = fmt(np.tile(utils.unsqueeze(obs, 0), (num, 1)))
        default = model.predict(obs)
        actions, action_dims = self._interp_actions(num, dims, default=default)
        actions = fmt(actions)
        assert states.size(0) == actions.size(0)
        assert states.size(1) == self.observation_space.shape[0]
        assert actions.size(1) == self.action_space.shape[0]

        # Infer action values
        q1s, q2s = model.network.critic(states, actions)
        qs = utils.to_np(torch.min(q1s, q2s))
        
        outputs = defaultdict(list)
        for state, action, action_dim in zip(utils.to_np(states), utils.to_np(actions), action_dims):
            outputs["state"].append(state)
            outputs["action"].append(action)
            outputs["action_dim"].append(action_dim)
            if not step: continue
            # Simulate forward cloned environments
            env = type(self).clone(self, **self._base_kwargs)
            outputs["env"].append(env)
            obs, rew, done, info = env.step(action)
            outputs["observation"].append(obs)
            outputs["reward"].append(rew)
            outputs["done"].append(done)
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
        assert issubclass(type(self), Box2DBase), "Must be called from a subclass of envs.pybox2d.base.Box2DBase"
        assert isinstance(model, Algorithm), "Model argument must be an instance of algs.Base.Algorithm"
        assert all(d in list(range(self.observation_space.shape[0])) for d in dims)
        
        # Tensorize states and actions
        fmt = model._format_batch
        default = self._get_observation()
        states, state_dims = self._interp_states(num, dims, default=default)
        states = fmt(states)
        actions = model.predict(states, is_batched=True)
        assert states.size(0) == actions.size(0)
        assert states.size(1) == self.observation_space.shape[0]
        assert actions.size(1) == self.action_space.shape[0]

        # Infer action values
        q1s, q2s = model.network.critic(states, actions)
        qs = utils.to_np(torch.min(q1s, q2s))

        outputs = defaultdict(list)
        for state, state_dim, action in zip(states, state_dims, actions):
            outputs["state"].append(state)
            outputs["state_dim"].append(state_dim)
            outputs["action"].append(action)
            if not step: continue
            # Simulate forward cloned environments
            env = type(self).clone(self, **self._base_kwargs)
            outputs["env"].append(env)
            obs, rew, done, info = env.step(action)
            outputs["observation"].append(obs)
            outputs["reward"].append(rew)
            outputs["done"].append(done)
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
        low, high = self.action_space.low, self.action_space.high
        if default is None: default = (low + high) * 0.5
        actions = np.linspace(low, high, num, dtype=np.float32)
        # Fill in default action values
        mask = np.ones_like(low, dtype=np.bool)
        mask[dims] = False
        actions[:, mask] = default[mask]
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
        low = self.observation_space.low
        high = self.observation_space.high
        if default is None: default = (low + high) * 0.5
        states = np.linspace(low, high, num, dtype=np.float32)
        # Fill in default state values
        mask = np.ones_like(low, dtype=np.bool)
        mask[dims] = False
        states[:, mask] = default[mask]
        assert all(self.observation_space.contains(s) for s in states)
        state_dims = states[:, mask].copy()
        return states, state_dims

    def _render_setup(self, mode="human"):
        """Initialize rendering parameters.
        """
        if mode == "human" or mode == "rgb_array":
            # Get image dimensions according the playground size
            workspace = self._get_shape_kwargs("playground")
            (w, h), t = workspace["size"], workspace["t"]
            r = 1 / t
            assert r.is_integer()
            y_range = int((w + 2*t) * r)
            x_range = int((h + t) * r)
            image = np.ones((x_range + 1, y_range + 1, 3), dtype=np.float32) * 255

            # Resolve reference frame transforms
            global_to_workspace = rigid_body_2d(0, -self._t_global[0], -self._t_global[1], r)
            workspace_to_image = rigid_body_2d(np.pi*0.5, h, w*0.5 + t, r)
            global_to_image = workspace_to_image @ global_to_workspace

            # Render static world bodies
            static_image = image.copy()
            for object_name in self.env.keys():
                if self._get_type(object_name) != "static": continue
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
            image_to_plot = rigid_body_2d(-np.pi*0.5, 0, image.shape[0])
            self._global_to_plot = image_to_plot @ global_to_image
            self._image = image
            self._static_image = static_image
            self._render_buffer = [self.render()]
    
    def render(self, mode="human", width=480, height=360):
        """Render sprites on all 2D bodies and set background to white.
        """        
        # Render all world shapes
        dynamic_image = self._image.copy()
        for object_name in self.env.keys():
            if self._get_type(object_name) == "static": continue
            for shape_name, shape_data in self._get_shapes(object_name).items():
                body = self._get_body(object_name, shape_name)
                position = np.array(body.position, dtype=np.float32)
                vertices = to_homogenous(shape_to_vertices(np.zeros_like(position), shape_data["box"])).T
                
                # Must orient vertices by angle about object centroid
                vertices = rigid_body_2d(body.angle, 0, 0) @ vertices
                vertices[:2, :] = (vertices[:2, :] + np.expand_dims(position, 1)) * self._r

                vertices_px = np.floor(self._global_to_image @ vertices).astype(int)
                x_idx, y_idx = draw.polygon(vertices_px[0, :], vertices_px[1, :], dynamic_image.shape)
                dynamic_image[x_idx, y_idx] = self._get_color(object_name)
        
        x_idx, y_idx = np.where(np.min(dynamic_image, axis=2) < 255)
        image = self._static_image.copy()
        image[x_idx, y_idx, :] = dynamic_image[x_idx, y_idx, :]
        image = np.round(image).astype(np.uint8)
        
        if mode == "human":
            image = self._render_util(image, width, height)
        return image

    def _render_util(self, image, width=480, height=360):
        """Caption and resize image.
        """
        image = Image.fromarray(image, "RGB")
        image = image.resize((width, height))
        caption = f"Env: {type(self).__name__} | Step: {self.steps} | "
        caption += f"Time: {self.physics_steps} | Reward: {self._cumulative_reward}"
        image = draw_caption(np.asarray(image), caption)
        return np.asarray(image, dtype=np.uint8)
