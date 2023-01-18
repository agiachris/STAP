import math
import torch
from torch import nn
from torch.nn import functional as F


class SquashedNormal(torch.distributions.TransformedDistribution):
    def __init__(self, loc, scale):
        self._loc = loc
        self.scale = scale
        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [torch.distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self):
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc


class MultiHeadAttention(nn.Module):
    """Vanilla multi-head self-attention later with project at the end."""

    def __init__(
        self, 
        n_head: int, 
        n_embd: int, 
        attn_pdrop: float, 
        resid_pdrop: float, 
    ):
        """Construct MultiHeadAttention.
        
        Args:
            n_head: Number of attention heads.
            n_embd: Number of embeddings.
            attn_pdrop: Attention dropout.
            resid_pdrop: Residual dropout.
        """
        super().__init__()
        assert n_embd % n_head == 0
        
        # Key, query, value projections for all heads.
        self._key = nn.Linear(n_embd, n_embd)
        self._query = nn.Linear(n_embd, n_embd)
        self._value = nn.Linear(n_embd, n_embd)
        
        # Regularization.
        self._attn_drop = nn.Dropout(attn_pdrop)
        self._resid_drop = nn.Dropout(resid_pdrop)

        # Output projection.
        self._proj = nn.Linear(n_embd, n_embd)
        self._n_head = n_head

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, T, C = q.size()
        _, S, C = k.size()

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
        q = self._query.forward(q).view(B, T, self._n_head, C // self._n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self._key.forward(k).view(B, S, self._n_head, C // self._n_head).transpose(1, 2)    # (B, nh, S, hs)
        v = self._value.forward(v).view(B, S, self._n_head, C // self._n_head).transpose(1, 2)  # (B, nh, S, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, S) -> (B, nh, T, S).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = F.softmax(att, dim=-1)
        att = self._attn_drop.forward(att)
        y = att @ v  # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side.

        # Output projection.
        y = self._resid_drop(self._proj(y))
        return y
