import gym
import torch

from temporal_policies.networks.actors.base import Actor
from temporal_policies.networks.random import Random


class RandomActor(Actor):
    """Dummy actor that returns random actions."""

    def __init__(self, action_space: gym.spaces.Box, dim_states: int):
        """Constructs the random actor.

        Args:
            action_space: Policy action space.
            dim_states: Dimensions of the input states.
        """
        super().__init__()
        self.network = Random(action_space.low, action_space.high, input_dim=dim_states)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """Outputs a random action.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        return self.network(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs a random action.

        Args:
            state: Environment state.

        Returns:
            Action.
        """
        return self.network.predict(state)
