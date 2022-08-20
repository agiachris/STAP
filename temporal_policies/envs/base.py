import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np


from temporal_policies.utils import spaces


class Primitive:
    action_space: gym.spaces.Box
    action_scale: gym.spaces.Box

    def __init__(self, idx_policy: int, policy_args):
        self._idx_policy = idx_policy
        self._policy_args = policy_args

    @property
    def idx_policy(self) -> int:
        return self._idx_policy

    @property
    def policy_args(self):
        return self._policy_args

    @classmethod
    def scale_action(cls, action: np.ndarray) -> np.ndarray:
        return spaces.transform(
            action, from_space=cls.action_space, to_space=cls.action_scale
        )

    @classmethod
    def normalize_action(cls, action: np.ndarray) -> np.ndarray:
        return spaces.transform(
            action, from_space=cls.action_scale, to_space=cls.action_space
        )

    def sample(self) -> np.ndarray:
        return self.action_space.sample()

    def __str__(self) -> str:
        args = "" if self.policy_args is None else ", ".join(map(str, self.policy_args))
        return f"{type(self).__name__}({args})"


class Env(gym.Env[np.ndarray, np.ndarray]):
    """Base env class with a separate state space for dynamics."""

    name: str
    observation_space: gym.spaces.Box
    state_space: gym.spaces.Box
    image_space: gym.spaces.Box

    render_mode: str  # type: ignore
    reward_range = (0.0, 1.0)

    @property
    def action_space(self) -> gym.spaces.Box:  # type: ignore
        return self.get_primitive().action_space

    @property
    def action_scale(self) -> gym.spaces.Box:
        return self.get_primitive().action_scale

    @property
    def primitives(self) -> List[str]:
        """Set of all environment primitive names."""
        raise NotImplementedError

    def get_primitive(self) -> Primitive:
        """Gets the environment primitive."""
        raise NotImplementedError

    def set_primitive(
        self,
        primitive: Optional[Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> "Env":
        """Sets the environment primitive."""
        raise NotImplementedError

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> Primitive:
        """Gets the primitive info."""
        raise NotImplementedError

    def create_primitive_env(self, primitive: Primitive) -> "Env":
        """Creates an child env with a fixed primitive mode."""
        return PrimitiveEnv(self, primitive)

    # @abc.abstractmethod
    # def create_policy_env(self, idx_policy: int, policy_args: Optional[Any]) -> "Env":
    #     raise NotImplementedError

    def get_state(self) -> np.ndarray:
        """Gets the environment state."""
        raise NotImplementedError

    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""
        raise NotImplementedError

    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        """Gets an observation for the current environment state."""
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError

    def render(self) -> np.ndarray:  # type: ignore
        raise NotImplementedError

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
            mode: Recording mode.
        Returns:
            Whether recording was started.
        """
        return False

    def record_stop(self, save_id: Optional[Any] = None, mode: str = "default") -> bool:
        """Stops recording the simulation.

        Args:
            save_id: Saves the recording to this id.
            mode: Recording mode.
        Returns:
            Whether recording was stopped.
        """
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
        return False


class PrimitiveEnv(Env):
    class Scope:
        def __init__(self, primitive_env: "PrimitiveEnv"):
            self._primitive_env = primitive_env
            self._env = primitive_env._env

        def __enter__(self):
            self._primitive = self._env.get_primitive()
            self._env.set_primitive(self._primitive_env._primitive)

        def __exit__(self, type, value, traceback):
            self._env.set_primitive(self._primitive)

    def __init__(self, env: Env, primitive: Primitive):
        self.name = f"{env.name}-{primitive}"
        self._env = env
        self._primitive = primitive

    def get_primitive(self) -> Primitive:
        return self._primitive

    def set_primitive(
        self,
        primitive: Optional[Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> Env:
        if primitive is None:
            primitive = self.get_primitive_info(action_call, idx_policy, policy_args)
        if primitive != self._primitive:
            raise ValueError("Primitive cannot be set for PrimitiveTableEnv")

        return self

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> Primitive:
        return self._env.get_primitive_info(action_call, idx_policy, policy_args)

    def create_primitive_env(self, primitive: Primitive) -> Env:
        if primitive != self._primitive:
            raise ValueError("Primitive env cannot created from PrimitiveTableEnv")
        return self

    def get_state(self) -> np.ndarray:
        with PrimitiveEnv.Scope(self):
            return self._env.get_state()

    def set_state(self, state: np.ndarray) -> bool:
        with PrimitiveEnv.Scope(self):
            return self._env.set_state(state)

    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        with PrimitiveEnv.Scope(self):
            return self._env.get_observation(image)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        with PrimitiveEnv.Scope(self):
            return self._env.reset(
                seed=seed,
                return_info=return_info,
                options=options,
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        with PrimitiveEnv.Scope(self):
            return self._env.step(action)

    def render(self) -> np.ndarray:  # type: ignore
        with PrimitiveEnv.Scope(self):
            return self.render()

    def record_start(
        self,
        prepend_id: Optional[Any] = None,
        frequency: Optional[int] = None,
        mode: str = "default",
    ) -> bool:
        return self._env.record_start(prepend_id, frequency, mode)

    def record_stop(self, save_id: Optional[Any] = None, mode: str = "default") -> bool:
        return self._env.record_stop(save_id, mode)

    def record_save(
        self,
        path: Union[str, pathlib.Path],
        reset: bool = True,
        mode: Optional[str] = None,
    ) -> bool:
        return self._env.record_save(path, reset, mode)


# class ProcessEnv(Env):
#     """Creates the env in a separate process."""
#
#     def __init__(self, env_class: Type[Env], env_kwargs: Dict[str, Any]):
#         parent_conn, child_conn = multiprocessing.Pipe()
#         self._conn = parent_conn
#
#         self._process = multiprocessing.Process(
#             target=ProcessEnv._run_env_process, args=(child_conn, env_class, env_kwargs)
#         )
#         self._process.start()
#
#         self._name = self._call("name")
#         self.state_space = self._call("state_space")
#         self.action_space = self._call("action_space")
#         self.observation_space = self._call("observation_space")
#
#     def _call(self, method: str, *args, **kwargs):
#         self._conn.send((method, args, kwargs))
#         return self._conn.recv()
#
#     def get_state(self) -> np.ndarray:
#         return self._call("get_state")
#
#     def set_state(self, state: np.ndarray) -> bool:
#         return self._call("set_state")
#
#     def get_observation(self) -> np.ndarray:
#         return self._call("get_observation")
#
#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         return self._call("step", action)
#
#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         return_info: bool = False,
#         options: Optional[dict] = None
#     ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
#         return self._call("reset", seed=seed, return_info=return_info, options=options)
#
#     def render(self, mode: str = "human"):
#         return self._call("render", mode)
#
#     def close(self):
#         result = self._call("close")
#         self._process.join()
#         return result
#
#     def __del__(self):
#         self.close()
#         super().__del__()
#
#     def _run_env_process(
#         conn: multiprocessing.connection.Connection,
#         env_class: Type[Env],
#         env_kwargs: Dict[str, Any],
#     ) -> None:
#         env = env_class(**env_kwargs)
#         while True:
#             method, args, kwargs = conn.recv()
#
#             attr = getattr(env, method)
#             if inspect.ismethod(attr):
#                 result = attr(*args, **kwargs)
#             else:
#                 result = attr
#
#             conn.send(result)
#
#             if method == "close":
#                 break
