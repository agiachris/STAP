import abc
import math
from typing import Optional, Union, Tuple, Callable, Type

import torch
from torch import nn

from temporal_policies import scod
from temporal_policies.utils import tensors


class WrapperSCOD(scod.SCOD, abc.ABC):
    """Base wrapper class for the SCOD class."""

    def __init__(
        self,
        model: nn.Module,
        output_agg_func: Optional[Union[str, Callable]] = None,
        sketch_cls: Type[scod.SinglePassPCA] = scod.SinglePassPCA,
        num_eigs: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[str] = None,
    ):
        """Construct the SCOD wrapper."""
        super().__init__(
            model,
            output_agg_func=output_agg_func,
            sketch_cls=sketch_cls,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        # Critic outputs a float32 scalar.
        OUTPUT_SIZE = 1
        OUTPUT_DTYPE = torch.float32

        # Estimate the rough amount of memory required for one element.
        # Assume float32 parameters.
        element_size = self._num_params * OUTPUT_SIZE * 4

        # Preallocate results.
        batch_size = input[0].shape[0]
        outputs = torch.zeros(
            batch_size, OUTPUT_SIZE, dtype=OUTPUT_DTYPE, device=self.device
        )
        if mode == 2 or mode == 3:
            variances = None
        else:
            variances = torch.zeros(batch_size, OUTPUT_SIZE, device=self.device)
        if mode == 1 or mode == 3:
            uncertainties = None
        else:
            uncertainties = torch.zeros(batch_size, 1, device=self.device)

        if self.device.type == "cuda":
            minibatch_size, num_minibatches = tensors.compute_minibatch(batch_size, element_size)
        else:
            # If on CPU, keep the minibatch size at a reasonable size.
            minibatch_size = min(batch_size, 10000)

        # Query SCOD in minibatches.
        for i in range(int(math.ceil(batch_size / minibatch_size) + 0.5)):
            idx_start = i * minibatch_size
            idx_end = idx_start + minibatch_size

            minibatch = tuple(x[idx_start:idx_end] for x in input)
            output, variance, uncertainty = super().forward(
                minibatch, detach=detach, mode=mode
            )

            outputs[idx_start:idx_end] = output
            if variances is not None:
                variances[idx_start:idx_end] = variance
            if uncertainties is not None:
                uncertainties[idx_start:idx_end] = uncertainty

        return outputs, variances, uncertainties

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
