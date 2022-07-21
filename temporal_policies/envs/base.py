import abc
import pathlib
from typing import Any, Optional, Union

import gym
import numpy as np


from temporal_policies.utils import spaces


class Primitive(abc.ABC):
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

    def normalize_action(cls, action: np.ndarray) -> np.ndarray:
        return spaces.transform(
            action, from_space=cls.action_scale, to_space=cls.action_space
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self.policy_args))})"


class Env(gym.Env[np.ndarray, np.ndarray]):
    """Base env class with a separate state space for dynamics."""

    name: str
    observation_space: gym.spaces.Box
    state_space: gym.spaces.Box
    image_space: gym.spaces.Box

    @property
    def action_space(self) -> gym.spaces.Box:  # type: ignore
        return self.get_primitive().action_space

    @property
    def action_scale(self) -> gym.spaces.Box:
        return self.get_primitive().action_scale

    @abc.abstractmethod
    def get_primitive(self) -> Primitive:
        """Gets the environment primitive."""

    @abc.abstractmethod
    def set_primitive(
        self,
        primitive: Optional[Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> "Env":
        """Sets the environment primitive."""

    @abc.abstractmethod
    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> Primitive:
        """Gets the primitive info."""

    # @abc.abstractmethod
    # def create_policy_env(self, idx_policy: int, policy_args: Optional[Any]) -> "Env":
    #     raise NotImplementedError

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """Gets the environment state."""

    @abc.abstractmethod
    def set_state(self, state: np.ndarray) -> bool:
        """Sets the environment state."""

    @abc.abstractmethod
    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        """Gets an observation for the current environment state."""

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
#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
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
