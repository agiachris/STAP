import torch  # type: ignore


class Random(torch.nn.Module):
    """Outputs a uniformly sampled random value."""

    def __init__(self, min: torch.Tensor, max: torch.Tensor, input_dim: int = 1):
        """Constructs the random network.

        Args:
            min: Minimum output.
            max: Maximum output.
            input_dim: Dimensions of the network's first input.
        """
        super().__init__()
        self.min = min
        self.scale = max - min
        self.dim = input_dim

    def _apply(self, fn):
        super()._apply(fn)
        self.min = fn(self.min)
        self.scale = fn(self.scale)
        return self

    def forward(self, input: torch.Tensor, *args) -> torch.Tensor:
        """Outputs a random value according to the input batch dimensions.

        Args:
            input: First network input.
        """
        shape = input.shape[:-self.dim] if self.dim > 0 else input.shape

        random = torch.rand(*shape, *self.min.shape)
        scaled = self.scale * random + self.min

        return scaled
