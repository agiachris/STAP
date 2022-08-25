import torch

from temporal_policies import scod


class CoVSCOD(scod.WrapperSCOD):
    """A SCOD wrapper computing the coefficient of variation of the posterior
    predictive distribution."""

    def predict(
        self,
        *input: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Compute the coefficient of variation of the posterior predictive outputs.

        Args:
            input: Model inputs of shape (B x d_in)
            detach: Remove jacobians and model outputs from the computation graph (default: True)

        Returns:
            metric: Posterior predictive coefficienct of variation (B) or (B x d_out)
        """
        output, variance, _ = self.forward(*input, detach=detach)
        assert variance is not None
        variance = (variance - variance.min()) / (variance.max() - variance.min())
        metric = output / variance
        if metric.size(-1) == 1:
            metric = metric.squeeze(-1)
        return metric
