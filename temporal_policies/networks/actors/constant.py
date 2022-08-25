from typing import Sequence, Union

import numpy as np
import torch

from temporal_policies.networks.actors.base import Actor
from temporal_policies.networks.constant import Constant


class ConstantActor(Actor):
    """Dummy actor that returns constant actions."""

    def __init__(
        self,
        constant: Union[torch.Tensor, np.ndarray, Sequence[float]],
        dim_states: int,
        dim_batch: int,
    ):
        """Constructs the random actor.

        Args:
            constant: Constant output.
            dim_states: Dimensions of the input state.
            dim_batch: Dimensions of the input batch.
        """
        super().__init__()
        self.network = Constant(constant, input_dim=dim_states + dim_batch)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """Outputs a constant action.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        return self.network(state)

    def predict(self, state: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Outputs a constant action.

        Args:
            state: Environment state.
            sample: Should always be false for ConstantActor.

        Returns:
            Action.
        """
        return self.network.predict(state)
