from typing import Optional

import gym

from temporal_policies.encoders.base import Encoder
from temporal_policies import envs, networks


class IdentityEncoder(Encoder):
    """Dummy encooder."""

    def __init__(
        self,
        env: Optional[envs.Env],
        action_space: Optional[gym.spaces.Box],
        observation_space: Optional[gym.spaces.Box],
    ):
        """Initializes the identity encoder.

        Args:
            env: Encoder env.
        """
        if action_space is None and env is not None:
            action_space = env.action_space
        if observation_space is None and env is not None:
            observation_space = env.observation_space
        if action_space is None or observation_space is None:
            raise ValueError(
                "Either env or (action_space, observation_space) must not be None."
            )

        if env is None:
            env = envs.EmptyEnv(
                observation_low=observation_space.low,
                observation_high=observation_space.high,
                observation_shape=observation_space.shape,
                observation_dtype=observation_space.dtype,
                action_low=action_space.low,
                action_high=action_space.high,
                action_shape=action_space.shape,
                action_dtype=action_space.dtype,
            )

        super().__init__(env, networks.encoders.IdentityEncoder)
