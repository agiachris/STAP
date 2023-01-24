from typing import List, Optional, Type

import torch

from temporal_policies.networks.critics.base import Critic
from temporal_policies.networks.transformer import AttentionEncoderBlock


def create_q_network(
    observation_space,
    action_space,
    attention_heads: int, 
    embedding_size: int,
    attention_dropout: float,
    residual_dropout: float,
    residual_ratio: int,
    output_act: Optional[Type[torch.nn.Module]] = None,
) -> torch.nn.Module:
    
    attention = AttentionEncoderBlock(
        attention_heads, 
        embedding_size, 
        attention_dropout, 
        residual_dropout
    )
    return None



class ContinuousTransformerCritic(Critic):
    def __init__(
        self,
        observation_space,
        action_space,
        attention_heads: int, 
        embedding_size: int,
        attention_dropout: float,
        residual_dropout: float,
        residual_ratio: int,
        num_q_fns: int = 2,
        output_act: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__()

        self.qs = torch.nn.ModuleList(
            [
                create_q_network(
                    observation_space,
                    action_space,
                    hidden_layers,
                    act,
                    fourier_features,
                    output_act,
                )
                for _ in range(num_q_fns)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        x = torch.cat((state, action), dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        qs = self.forward(state, action)
        return torch.min(*qs)
