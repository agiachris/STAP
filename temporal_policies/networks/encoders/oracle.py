from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from temporal_policies import envs
from temporal_policies.networks.encoders.base import Decoder, Encoder
from temporal_policies.utils import tensors


class OracleEncoder(Encoder):
    """Dummy encoder that returns the ground truth environment state.

    For use with OracleAgent and OracleDynamics.
    """

    def __init__(self, env: envs.Env):
        """Constructs the oracle encoder.

        Args:
            env: Gym environment.
        """
        super().__init__(env, env.state_space)

        self.env = env

        if not hasattr(self.env, "_state_obs_cache"):
            # Can't cache obs->state because observations are not unique to a state.
            self.env._state_obs_cache: Dict[Tuple[str, Tuple], np.ndarray] = {}  # type: ignore

        @tensors.vmap(dims=len(self.env.observation_space.shape))
        def forward(observation: np.ndarray) -> np.ndarray:
            env_observation = self.env.get_observation()
            assert observation.ndim == env_observation.ndim
            if (observation != env_observation).any():
                # May happen if self.env is not updated by the dynamics factory.
                raise ValueError("Observation does not match the current env state")

            state = self.env.get_state()

            # Save in cache for decoder.
            self.env._state_obs_cache[  # type: ignore
                str(self.env.get_primitive()), tuple(state)
            ] = observation

            return state

        self._forward = forward

    def reset_cache(self) -> None:
        self.env._state_obs_cache.clear()  # type: ignore

    @tensors.torch_wrap
    def forward(
        self, observation: np.ndarray, policy_args: Union[np.ndarray, Optional[Any]]
    ) -> np.ndarray:
        """Returns the current environment state.

        WARNING: This ignores the input observation and instead returns the
        environment's current ground truth state. Be careful that the state
        matches the observation as expected.
        """
        return self._forward(observation)


class OracleDecoder(Decoder):
    """Dummy decoder that returns an observation from the ground truth environment state.

    For use with OracleActor and OracleDynamics.
    """

    def __init__(self, env: envs.Env):
        """Constructs the oracle encoder.

        Args:
            env: Gym environment.
        """
        super().__init__(env)

        self.env = env

        if not hasattr(self.env, "_state_obs_cache"):
            self.env._state_obs_cache: Dict[Tuple[str, Tuple], np.ndarray] = {}  # type: ignore

        @tensors.vmap(dims=len(self.env.state_space.shape))
        def forward(state: np.ndarray) -> np.ndarray:
            try:
                # Get the observation from cache.
                observation = self.env._state_obs_cache[  # type: ignore
                    str(self.env.get_primitive()), tuple(state)
                ]
            except KeyError:
                prev_state = self.env.get_state()
                self.env.set_state(state)

                observation = self.env.get_observation()

                self.env.set_state(prev_state)

                # Save in cache.
                self.env._state_obs_cache[  # type: ignore
                    str(self.env.get_primitive()), tuple(state)
                ] = observation

            return observation

        self._forward = forward

    def reset_cache(self) -> None:
        self.env._state_obs_cache.clear()  # type: ignore

    @tensors.torch_wrap
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Decodes the environment state into an observation.

        WARNING: This returns an observation according to the current env
        primitive. Be careful that the primitive is properly set.
        """
        return self._forward(state)
