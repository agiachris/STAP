import torch
import torch.nn as nn

from temporal_policies.networks.utils import MultiHeadAttention


class AttentionEncoderBlock(nn.Module):
    """Encoder transformer block."""

    def __init__(
        self,
        attention_heads: int,
        embedding_size: int,
        attention_dropout: float,
        residual_dropout: float,
        residual_ratio: int,
    ):
        """Construct attention encoder block.

        Args:
            attention
        """
        super().__init__()
        self._layer_norm = nn.LayerNorm(embedding_size)
        self._attention = MultiHeadAttention(
            attention_heads, embedding_size, attention_dropout, residual_dropout
        )
        self._residual = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, residual_ratio * embedding_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(residual_ratio * embedding_size, embedding_size),
            nn.Dropout(residual_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln_x = self._layer_norm.forward(x)
        x = x + self._attention.forward(ln_x, ln_x, ln_x)
        return x + self._residual.forward(x)
