from typing import List

import torch  # type: ignore

from temporal_policies.networks.critics.base import Critic
from temporal_policies.networks.mlp import MLP, weight_init


class ContinuousMLPCritic(Critic):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers=[256, 256],
        act=torch.nn.ReLU,
        num_q_fns=2,
        ortho_init=False,
    ):
        super().__init__()
        self.qs = torch.nn.ModuleList(
            [
                MLP(
                    observation_space.shape[0] + action_space.shape[0],
                    1,
                    hidden_layers=hidden_layers,
                    act=act,
                )
                for _ in range(num_q_fns)
            ]
        )
        if ortho_init:
            self.apply(weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
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
