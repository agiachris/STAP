from typing import Any

import torch
import scipy.stats as stats

from temporal_policies import scod


class VaRSCOD(scod.WrapperSCOD):
    """SCOD Wrapper computing the parametric value-at-risk (VaR) over a
    Gaussian-distributed posterior predictive Q-value."""

    def __init__(self, alpha: float = 0.9, **kwargs: Any):
        """Constructs VaRSCOD.

        Args:
            alpha: VaR risk-aversion parameter (confidence percentile)
        """
        super().__init__(**kwargs)
        assert (
            0.0 < alpha < 1
        ), "VaR confidence percentile (alpha) must be between (0, 1)"
        self._alpha = alpha
        self._zscore = torch.tensor(stats.norm.ppf(1 - self.alpha)).to(self.device)

    @property
    def alpha(self) -> float:
        """VaR confidence percentile."""
        return self._alpha

    @property
    def zscore(self) -> torch.Tensor:
        """Lower bound VaR zscore corresponding to the set confidence percentile."""
        return self._zscore

    def predict(
        self,
        *input: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Compute alpha-averse confidence lower-bound over the posterior predictive outputs.

        Args:
            input: Tensor or sequence of model inputs of shape (B x d_in)
            detach: Remove jacobians and model outputs from the computation graph (default: True)

        Returns:
            metric: Posterior predictive confidence lower-bound (B) or (B x d_out)
        """
        output, variance, _ = self.forward(*input, detach=detach)
        assert variance is not None
        variance = (variance - variance.min()) / (variance.max() - variance.min())
        metric = output + variance * self.zscore
        if metric.size(-1) == 1:
            metric = metric.squeeze(-1)
        return metric
