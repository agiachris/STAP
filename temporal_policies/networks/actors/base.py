import abc

import torch


class Actor(torch.nn.Module, abc.ABC):
    """Base actor class."""

    @abc.abstractmethod
    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """Outputs the actor distribution.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        pass

    @abc.abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs the actor prediction.

        Args:
            state: Environment state.

        Returns:
            Action.
        """
        pass
