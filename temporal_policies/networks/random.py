from typing import Union

import numpy as np
import torch

from temporal_policies.utils import tensors


class Random(torch.nn.Module):
    """Outputs a uniformly sampled random value."""

    def __init__(
        self,
        min: Union[torch.Tensor, np.ndarray, float, int],
        max: Union[torch.Tensor, np.ndarray, float, int],
        input_dim: int = 1,
    ):
        """Constructs the random network.

        Args:
            min: Minimum output.
            max: Maximum output.
            input_dim: Dimensions of the network's first input.
        """
        super().__init__()
        self.min = tensors.to_tensor(min)
        self.scale = tensors.to_tensor(max - min)
        self.dim = input_dim

    def _apply(self, fn):
        super()._apply(fn)
        self.min = fn(self.min)
        self.scale = fn(self.scale)
        return self

    def forward(self, input: torch.Tensor, *args) -> torch.Tensor:
        """Outputs a random value according to the input batch dimensions.

        Args:
            input: First network input.
        """
        shape = input.shape[: -self.dim] if self.dim > 0 else input.shape

        random = torch.rand(*shape, *self.min.shape, device=input.device)
        scaled = self.scale * random + self.min

        return scaled

    def predict(self, *args) -> torch.Tensor:
        return self.forward(*args)
