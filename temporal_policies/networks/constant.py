from typing import Sequence, Union

import numpy as np
import torch

from temporal_policies.utils import tensors


class Constant(torch.nn.Module):
    """Outputs a constant value."""

    def __init__(
        self,
        constant: Union[
            torch.Tensor, np.ndarray, Sequence[float], Sequence[int], float, int
        ],
        input_dim: int = 1,
    ):
        """Constructs the constant network.

        Args:
            constant: Constant output.
            input_dim: Dimensions of the network's first input.
        """
        super().__init__()
        self.constant = tensors.to_tensor(constant)
        self.dim = input_dim

    def _apply(self, fn):
        super()._apply(fn)
        self.constant = fn(self.constant)
        return self

    def forward(self, input: torch.Tensor, *args) -> torch.Tensor:
        """Outputs the constant repeated according to the input batch dimensions.

        Args:
            input: First network input.
        """
        if input.dim() == self.dim:
            return self.constant

        shape = input.shape[: -self.dim] if self.dim > 0 else input.shape

        if self.constant.dim() == 0:
            return self.constant.expand(*shape)

        return self.constant.expand(*shape, -1)

    def predict(self, input: torch.Tensor, *args) -> torch.Tensor:
        """Outputs the constant repeated according to the input batch dimensions.

        Args:
            input: First network input.
        """
        return self.forward(input, *args)
