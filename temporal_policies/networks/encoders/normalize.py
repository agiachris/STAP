import gym  # type: ignore
import torch  # type: ignore

from temporal_policies.networks.encoders import Encoder


class NormalizeObservation(Encoder):
    """Normalizes observation to the range (-0.5, 0.5)."""

    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__()
        self.observation_mid = torch.from_numpy(
            (observation_space.low + observation_space.high) / 2
        )
        self.observation_range = torch.from_numpy(
            observation_space.high - observation_space.low
        )

    def _apply(self, fn):
        """Ensures members get transferred with NormalizeObservation.to(device)."""
        super()._apply(fn)
        self.observation_mid = fn(self.observation_mid)
        self.observation_range = fn(self.observation_range)
        return self

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalizes observation to the range (-0.5, 0.5)."""
        return (observation - self.observation_mid) / self.observation_range
