import abc

import torch


class ProbabilisticCritic(torch.nn.Module, abc.ABC):
    """Probabilistic critic class."""

    @abc.abstractmethod
    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Predicts the output distribution of a (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted probability distribution.
        """
        pass

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        pass
