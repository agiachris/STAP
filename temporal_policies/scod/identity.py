from typing import Any, List, Union

import torch

from temporal_policies import scod


class IdentitySCOD(scod.WrapperSCOD):
    """SCOD wrapper returning the model output."""

    def __init__(self, **kwargs: Any):
        """Constructs IdentitySCOD."""
        super().__init__(**kwargs)

    def predict(
        self,
        *input: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Compute model output without application of the SCOD produced
        variance or uncertainty metrics.
        """
        output, _, _ = self.forward(*input, detach=detach, mode=3)
        if output.size(-1) == 1:
            output = output.squeeze(-1)
        return output
