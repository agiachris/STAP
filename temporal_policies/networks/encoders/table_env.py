from typing import Optional

import gym
import numpy as np
import torch

from temporal_policies import envs
from temporal_policies.networks.encoders import Encoder
from temporal_policies.utils import tensors


class TableEnvEncoder(Encoder):
    """Encoder for TableEnv observations.

    Converts the 2D low-dimensional state returned by TableEnv into a 1D vector
    for the primitive policies. The objects are re-ordered such that the first
    corresponds to the gripper, the second/third correspond to the primitive
    argument objects, and the remaining are a random permutation of objects.

    The observation values are also normalized to the range (-0.5, 0.5).
    """

    def __init__(self, env: envs.pybullet.TableEnv):
        observation_space = env.observation_space

        state_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(int(np.prod(observation_space.shape)),),
            dtype=np.float32,
        )
        super().__init__(env, state_space)

        self.observation_mid = torch.from_numpy(
            (observation_space.low[0] + observation_space.high[0]) / 2
        )
        self.observation_range = torch.from_numpy(
            observation_space.high[0] - observation_space.low[0]
        )

        self._env = env
        # self._idx_args: Optional[List[int]] = None

    def _apply(self, fn):
        """Ensures members get transferred with TableEnvEncoder.to(device)."""
        super()._apply(fn)
        self.observation_mid = fn(self.observation_mid)
        self.observation_range = fn(self.observation_range)
        return self

    @tensors.batch(dims=2)
    def forward(
        self,
        observation: torch.Tensor,
        env: Optional[envs.pybullet.TableEnv] = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """Encodes the TableEnv observation into a flat vector.

        Converts the 2D low-dimensional state returned by TableEnv into a 1D
        vector for the primitive policies. The objects are re-ordered such that
        the first corresponds to the gripper, the second/third correspond to the
        primitive argument objects, and the remaining are a random permutation
        of objects.

        The observation values are also normalized to the range (-0.5, 0.5).

        Args:
            observation: TableEnv observation.
            env: Optional TableEnv, for example if an eval env should be used
                instead of the default train env.
            randomize: Whether to randomize the order of auxiliary objects.
        """
        if env is None:
            env = self._env
        primitive = env.get_primitive()
        idx_args, num_objects = env.get_arg_indices(
            idx_policy=primitive.idx_policy, policy_args=primitive.policy_args
        )

        if randomize:
            np_idx_args = np.array(idx_args)
            np.random.shuffle(np_idx_args[1 + len(primitive.policy_args) : num_objects])
            idx_args = np_idx_args.tolist()

        observation = observation[:, idx_args, :]
        observation = (observation - self.observation_mid) / self.observation_range
        observation = torch.reshape(observation, (-1, self.state_space.shape[0]))

        return observation
