import abc

import torch  # type: ignore


class Encoder(torch.nn.Module, abc.ABC):
    """Base critic class."""

    @abc.abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encodes the observation to the policy latent state.

        Args:
            observation: Environment observation.

        Returns:
            Encoded policy state.
        """
        pass
