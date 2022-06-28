import abc
import inspect
import multiprocessing
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, Union

import gym  # type: ignore

from temporal_policies.utils.typing import StateType, ObsType, ActType


class Env(gym.Env[ObsType, ActType], Generic[StateType, ActType, ObsType]):
    """Base env class with a separate state space for dynamics."""

    name: str
    state_space: gym.spaces.Space[StateType]
    image_space: gym.spaces.Space[ObsType]

    @abc.abstractmethod
    def get_state(self) -> StateType:
        """Gets the environment state."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state: StateType) -> bool:
        """Sets the environment state."""
        raise NotImplementedError

    # TODO: Set idx_policy to first arg.
    @abc.abstractmethod
    def get_observation(self, image: Optional[bool] = None) -> ObsType:
        """Gets an observation for the current environment state."""
        raise NotImplementedError


class SequentialEnv(
    Env[StateType, Tuple[ActType, int, Any], ObsType],
    Generic[StateType, ActType, ObsType],
):
    """Wrapper around a sequence of child envs."""

    def __init__(self, envs: List[Env]):
        self._envs = envs
        self.state_space = self.envs[0].state_space
        self.name = "_".join([env.name for env in self.envs])

    @property
    def envs(self) -> List[Env]:
        """Primtive envs."""
        return self._envs

    def get_state(self) -> StateType:
        """Gets the environment state."""
        base_env = self.envs[0]
        return base_env.get_state()

    def set_state(self, state: StateType) -> bool:
        """Sets the environment state."""
        base_env = self.envs[0]
        return base_env.set_state(state)

    def get_observation(self, idx_policy: Optional[int] = None) -> ObsType:
        """Gets an observation for the current state of the environment."""
        if idx_policy is None:
            idx_policy = 0
        return self.envs[idx_policy].get_observation()

    def step(
        self, action: Tuple[ActType, int, Any]
    ) -> Tuple[ObsType, float, bool, Dict]:
        """Executes the step corresponding to the policy index.

        Args:
            action: 3-tuple (action, idx_policy, policy_args).

        Returns:
            4-tuple (observation, reward, done, info).
        """
        env_action, idx_policy, policy_args = action
        return self.envs[idx_policy].step(env_action)

    def reset(self, idx_policy: int) -> ObsType:
        return self.envs[idx_policy].reset()


class ProcessEnv(Env):
    """Creates the env in a separate process."""

    def __init__(self, env_class: Type[Env], env_kwargs: Dict[str, Any]):
        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn

        self._process = multiprocessing.Process(
            target=ProcessEnv._run_env_process, args=(child_conn, env_class, env_kwargs)
        )
        self._process.start()

        self._name = self._call("name")
        self.state_space = self._call("state_space")
        self.action_space = self._call("action_space")
        self.observation_space = self._call("observation_space")

    def _call(self, method: str, *args, **kwargs):
        self._conn.send((method, args, kwargs))
        return self._conn.recv()

    def get_state(self) -> StateType:
        return self._call("get_state")

    def set_state(self, state: StateType) -> bool:
        return self._call("set_state")

    def get_observation(self) -> ObsType:
        return self._call("get_observation")

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict]:
        return self._call("step", action)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[ObsType, Tuple[ObsType, Dict]]:
        return self._call("reset", seed=seed, return_info=return_info, options=options)

    def render(self, mode: str = "human"):
        return self._call("render", mode)

    def close(self):
        result = self._call("close")
        self._process.join()
        return result

    def __del__(self):
        self.close()
        super().__del__()

    def _run_env_process(
        conn: multiprocessing.connection.Connection,
        env_class: Type[Env],
        env_kwargs: Dict[str, Any],
    ) -> None:
        env = env_class(**env_kwargs)
        while True:
            method, args, kwargs = conn.recv()

            attr = getattr(env, method)
            if inspect.ismethod(attr):
                result = attr(*args, **kwargs)
            else:
                result = attr

            conn.send(result)

            if method == "close":
                break
