from typing import Optional, Union

import numpy as np
import torch

from temporal_policies.utils import tensors


class Gaussian(torch.nn.Module):
    """Outputs a normally sampled value."""

    def __init__(
        self,
        mean: torch.nn.Module,
        std: float,
        min: Optional[Union[torch.Tensor, np.ndarray, float, int]] = None,
        max: Optional[Union[torch.Tensor, np.ndarray, float, int]] = None,
    ):
        """Constructs the random network.

        Args:
            mean: Mean network.
            std: Scalar standard deviation.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.min = tensors.to_tensor(min)
        self.max = tensors.to_tensor(max)

    def _apply(self, fn):
        super()._apply(fn)
        if self.min is not None:
            self.min = fn(self.min)
        if self.max is not None:
            self.max = fn(self.max)
        return self

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Outputs a random value sampled the Gaussian."""
        mean = self.mean(*args, **kwargs)
        sample = torch.random(mean, self.std)
        if self.min is not None and self.max is not None:
            sample = torch.clamp(sample, self.min, self.max)
        return sample

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Outputs a random value sampled the Gaussian."""
        mean = self.mean.predict(*args, **kwargs)
        sample = torch.normal(mean, self.std)
        if self.min is not None and self.max is not None:
            sample = torch.clamp(sample, self.min, self.max)
        return sample
