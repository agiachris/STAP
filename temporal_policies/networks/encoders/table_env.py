from typing import Dict, List, Union

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

    def _apply(self, fn):
        """Ensures members get transferred with TableEnvEncoder.to(device)."""
        super()._apply(fn)
        self.observation_mid = fn(self.observation_mid)
        self.observation_range = fn(self.observation_range)
        return self

    @staticmethod
    @tensors.vmap(dims=0)
    def _get_observation_indices(
        policy_args: Dict[str, List[int]], randomize: bool
    ) -> np.ndarray:
        """Gets the observation indices from the policy_args dict and shuffles
        the indices inside the shuffle range."""
        observation_indices = np.array(policy_args["observation_indices"])

        if randomize:
            shuffle_range = policy_args["shuffle_range"]
            np.random.shuffle(observation_indices[shuffle_range[0] : shuffle_range[1]])

        return observation_indices

    @staticmethod
    def rearrange_observation(
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Dict[str, List[int]]],
        randomize: bool = False,
    ) -> torch.Tensor:
        """Rearranges the objects in the observation matrix so that the
        end-effector and primitive args are first.

        Args:
            observation: TableEnv observation.
            observation_indices: List of indices computed by `primitive.get_policy_args()`.

        Returns:
            Observation with rearranged objects.
        """
        observation_indices = TableEnvEncoder._get_observation_indices(
            policy_args, randomize=randomize
        )

        # [num_objects] or [B, num_objects].
        t_observation_indices = torch.from_numpy(observation_indices).to(
            observation.device
        )
        if t_observation_indices.dim() == 1:
            # [num_objects] => [B, num_objects].
            t_observation_indices = t_observation_indices.unsqueeze(0)
        # [B, num_objects] => [B, num_objects, 1].
        t_observation_indices = t_observation_indices.unsqueeze(-1)
        # [B, num_objects, 1] => [B, num_objects, object_state_size].
        t_observation_indices = t_observation_indices.expand(
            -1, -1, observation.shape[-1]
        )

        # [B, num_objects, object_state_size].
        observation = torch.gather(observation, dim=1, index=t_observation_indices)

        return observation

    @tensors.batch(dims=2)
    def forward(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Dict[str, List[int]]],
        randomize: bool = True,
        **kwargs,
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
            policy_args: Auxiliary policy arguments.
            randomize: Whether to randomize the order of auxiliary objects.
        """
        observation = TableEnvEncoder.rearrange_observation(
            observation, policy_args, randomize
        )

        # print("encoded:", observation)
        observation = (observation - self.observation_mid) / self.observation_range
        # [B, num_objects, object_state_size] => [B, num_objects * object_state_size].
        observation = torch.reshape(observation, (-1, self.state_space.shape[0]))

        return observation
