from typing import Optional

import gym
import torch

from stap import envs
from stap.networks.actors.base import Actor
from stap.networks.random import Random
from stap.utils import tensors


class RandomActor(Actor):
    """Dummy actor that returns random actions."""

    def __init__(
        self,
        action_space: gym.spaces.Box,
        dim_states: int,
    ):
        """Constructs the random actor.

        Args:
            action_space: If not None, this actor will sample uniformly from the
                action space.
            dim_states: Dimensions of the input states.
        """
        super().__init__()

        self.network = Random(action_space.low, action_space.high, input_dim=dim_states)

        @tensors.vmap(dim_states)
        def sample_primitive(state: torch.Tensor) -> torch.Tensor:
            return torch.from_numpy(self._primitive.sample()).to(state.device)  # type: ignore

        self._sample_primitive = sample_primitive
        self._primitive: Optional[envs.Primitive] = None

    def set_primitive(self, primitive: Optional[envs.Primitive]) -> None:
        self._primitive = primitive

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """Outputs a random action.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        if self._primitive is not None:
            return self._sample_primitive(state)
        else:
            return self.network(state)

    def predict(self, state: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Outputs a random action.

        Args:
            state: Environment state.
            sample: Should always be true for RandomActor.

        Returns:
            Action.
        """
        if self._primitive is not None:
            return self._sample_primitive(state)
        else:
            return self.network.predict(state)
