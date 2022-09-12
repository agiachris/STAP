from typing import List, Optional

import torch
import numpy as np

from temporal_policies.networks.critics.base import Critic
from temporal_policies.networks.mlp import MLP, weight_init


class LFF(torch.nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, in_features, out_features, scale=1.0, init="iso", sincos=False):
        super().__init__()
        self.in_features = in_features
        self.sincos = sincos
        self.out_features = out_features
        self.scale = scale
        if self.sincos:
            self.linear = torch.nn.Linear(in_features, self.out_features // 2)
        else:
            self.linear = torch.nn.Linear(in_features, self.out_features)
        if init == "iso":
            torch.nn.init.normal_(self.linear.weight, 0, scale / self.in_features)
            torch.nn.init.normal_(self.linear.bias, 0, 1)
        else:
            torch.nn.init.uniform_(
                self.linear.weight, -scale / self.in_features, scale / self.in_features
            )
            torch.nn.init.uniform_(self.linear.bias, -1, 1)
        if self.sincos:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, **_):
        x = np.pi * self.linear(x)
        if self.sincos:
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            return torch.sin(x)


def create_q_network(
    observation_space, action_space, hidden_layers, act, fourier_features: Optional[int]
) -> torch.nn.Module:
    if fourier_features is not None:
        lff = LFF(observation_space.shape[0] + action_space.shape[0], fourier_features)
        mlp = MLP(
            fourier_features,
            1,
            hidden_layers=hidden_layers,
            act=act,
        )
        return torch.nn.Sequential(lff, mlp)
    else:
        mlp = MLP(
            observation_space.shape[0] + action_space.shape[0],
            1,
            hidden_layers=hidden_layers,
            act=act,
        )
        return mlp


class ContinuousMLPCritic(Critic):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers=[256, 256],
        act=torch.nn.ReLU,
        num_q_fns=2,
        ortho_init=False,
        fourier_features: Optional[int] = None,
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
                )
                for _ in range(num_q_fns)
            ]
        )
        if ortho_init:
            self.apply(weight_init)

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
