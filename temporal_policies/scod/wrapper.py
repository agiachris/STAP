import abc
from typing import Optional, Union, Tuple, Callable

import torch
from torch import nn

from temporal_policies import scod


class WrapperSCOD(scod.SCOD, abc.ABC):
    """Base wrapper class for the SCOD class."""

    def __init__(
        self,
        model: nn.Module,
        output_agg_func: Optional[Union[str, Callable]] = None,
        num_eigs: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[str] = None,
    ):
        """Construct the SCOD wrapper."""
        super().__init__(
            model,
            output_agg_func=output_agg_func,
            num_eigs=num_eigs,
            device=device,
            checkpoint=checkpoint,
        )

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def forward(
        self,
        *input: torch.Tensor,
        detach: bool = True,
        mode: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute model outputs, posterior predictive variances and uncertainties.

        Args:
            input: Model inputs of shape (B x d_in)
            detach: Remove jacobians and model outputs from the computation graph (default: True)
            mode: Int defining the return uncertainty metrics from SCOD (default: 1)

        Returns: (
            output: Model outputs of shape (B x d_out)
            variance: Posterior predictive variance of shape (B x d_out)
            uncertainty: Posterior predictive KL-divergence (B x 1) (default: None)
        )
        """
        return super().forward(input, detach=detach, mode=mode)

    @abc.abstractmethod
    def predict(
        self,
        *input: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Compute custom output quantity from outputs, posterior predictive
        variances, and uncertaintites.

        Args:
            input: Model inputs of shape (B x d_in)
            detach: Remove jacobians and model outputs from the computation graph (default: True)

        Returns:
            metric: Uncertainty-derived metric of shape (B x 1)
        """
        raise NotImplementedError
