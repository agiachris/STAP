from typing import Optional, Sequence, Type

import gym
import torch

from temporal_policies.networks.mlp import LFF, MLP, weight_init
from temporal_policies.networks.actors import base
from temporal_policies.networks.utils import SquashedNormal


class ContinuousMLPActor(base.Actor):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_layers: Sequence[int] = [256, 256],
        act: Type[torch.nn.Module] = torch.nn.ReLU,
        output_act: Type[torch.nn.Module] = torch.nn.Tanh,
        ortho_init: bool = False,
    ):
        super().__init__()
        self.mlp = MLP(
            state_space.shape[0],
            action_space.shape[0],
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act,
        )
        if ortho_init:
            self.apply(weight_init)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        return self.mlp(state)

    def predict(self, state: torch.Tensor, sample: bool = False) -> torch.Tensor:
        return self.mlp(state)


class DiagonalGaussianMLPActor(base.Actor):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_layers: Sequence[int] = [256, 256],
        act: Type[torch.nn.Module] = torch.nn.ReLU,
        ortho_init: bool = False,
        log_std_bounds: Sequence[int] = [-5, 2],
        fourier_features: Optional[int] = None,
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        if log_std_bounds is not None:
            assert log_std_bounds[0] < log_std_bounds[1]
        if fourier_features is not None:
            lff = LFF(state_space.shape[0], fourier_features)
            mlp = MLP(
                fourier_features,
                2 * action_space.shape[0],
                hidden_layers=hidden_layers,
                act=act,
                output_act=None,
            )
            self.mlp: torch.nn.Module = torch.nn.Sequential(lff, mlp)
        else:
            self.mlp = MLP(
                state_space.shape[0],
                2 * action_space.shape[0],
                hidden_layers=hidden_layers,
                act=act,
                output_act=None,
            )
        if ortho_init:
            self.apply(weight_init)
        self.action_range = [
            float(action_space.low.min()),
            float(action_space.high.max()),
        ]

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mu, log_std = self.mlp(state).chunk(2, dim=-1)
        if self.log_std_bounds is not None:
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            dist_class: Type[torch.distributions.Distribution] = SquashedNormal
        else:
            dist_class = torch.distributions.Normal
        std = log_std.exp()
        dist = dist_class(mu, std)
        return dist

    def predict(self, state: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self(state)
        if sample:
            action = dist.sample()
        else:
            action = dist.loc
        action = action.clamp(*self.action_range)
        return action
