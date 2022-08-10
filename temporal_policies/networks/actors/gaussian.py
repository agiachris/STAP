import gym
import torch

from temporal_policies.networks.actors.base import Actor
from temporal_policies.networks.gaussian import Gaussian


class GaussianActor(Actor):
    """Wrapper actor that returns a gaussian centered around a policy's prediction."""

    def __init__(self, actor: Actor, std: float, action_space: gym.spaces.Box):
        """Constructs the random actor.

        Args:
            actor: Main actor whose predictions are used as the mean.
            std: Standard deviation.
            action_space: Policy action space.
        """
        super().__init__()

        # Scale the standard deviations by the action space.
        std = 0.5 * std * (action_space.high - action_space.low)
        self.network = Gaussian(actor, std, min=action_space.low, max=action_space.high)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """Outputs a random action.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        return self.network(state)

    def predict(self, state: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Outputs a random action.

        Args:
            state: Environment state.
            sample: Should always be true for GaussianActor.

        Returns:
            Action.
        """
        return self.network.predict(state)
