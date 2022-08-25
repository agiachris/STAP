from typing import List, Optional, Type

import gym
import torch

from temporal_policies.networks.dynamics.base import PolicyDynamics
from temporal_policies.networks.mlp import MLP, weight_init


class MLPDynamics(PolicyDynamics):
    """Basic MLP for the dynamics model that concatenates the latent vector and policy
    parameters as input.

    The actions are scaled to be in the range (-0.5, 0.5).
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_layers: List[int] = [256, 256],
        act: Type[torch.nn.Module] = torch.nn.ReLU,
        output_act: Optional[Type[torch.nn.Module]] = None,
        ortho_init: bool = False,
    ):
        super().__init__(state_space, action_space)
        dim_latent = state_space.shape[0]
        self.mlp = MLP(
            dim_latent + action_space.shape[0],
            dim_latent,
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act,
        )
        self.dim_latent = dim_latent
        if isinstance(action_space, gym.spaces.Box):
            self.action_mid = torch.tensor((action_space.low + action_space.high) / 2)
            self.action_range = torch.tensor(action_space.high - action_space.low)
            self.action_dim = action_space.shape[0]
        else:
            raise NotImplementedError()

        if ortho_init:
            self.apply(weight_init)

    def _apply(self, fn):
        super()._apply(fn)
        self.action_mid = fn(self.action_mid)
        self.action_range = fn(self.action_range)
        return self

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = (action[..., : self.action_dim] - self.action_mid) / self.action_range

        return self.mlp(torch.cat((latent, action), dim=-1))
