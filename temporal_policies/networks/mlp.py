from typing import List

import gym

import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from .common import MLP

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class ContinuousMLPCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, num_q_fns=2, ortho_init=False):
        super().__init__()
        self.qs = nn.ModuleList([
            MLP(observation_space.shape[0] + action_space.shape[0], 1, hidden_layers=hidden_layers, act=act)
         for _ in range(num_q_fns)])
        if ortho_init:
            self.apply(weight_init)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]

class ContinuousMLPActor(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, output_act=nn.Tanh, ortho_init=False):
        super().__init__()
        self.mlp = MLP(observation_space.shape[0], action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=output_act)
        if ortho_init:
            self.apply(weight_init)
        
    def forward(self, obs):
        return self.mlp(obs)


class MLPDynamics(nn.Module):
    """Basic MLP for the dynamics model that concatenates the latent vector and policy
    parameters as input.

    The actions are scaled to be in the range (-0.5, 0.5).
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        dim_latent: int,
        hidden_layers: List[int] = [256, 256],
        act: nn.Module = nn.ReLU,
        output_act: nn.Module = None,
        ortho_init: bool = False
    ):
        super().__init__()
        self.mlp = MLP(
            dim_latent + action_space.shape[0],
            dim_latent,
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act
        )
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

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        action = (action[..., :self.action_dim] - self.action_mid) / self.action_range

        return self.mlp(torch.cat((latent, action), dim=-1))


class SquashedNormal(distributions.TransformedDistribution):

    def __init__(self, loc, scale):
        self._loc = loc
        self.scale = scale
        self.base_dist = distributions.Normal(loc, scale)
        transforms = [distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self):
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc

class DiagonalGaussianMLPActor(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, ortho_init=False, log_std_bounds=[-5, 2]):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        if log_std_bounds is not None:
            assert log_std_bounds[0] < log_std_bounds[1]
        self.mlp = MLP(observation_space.shape[0], 2*action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=None)
        if ortho_init:
            self.apply(weight_init)
        self.action_range = [float(action_space.low.min()), float(action_space.high.max())]
        
    def forward(self, obs):
        mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        if self.log_std_bounds is not None:
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            dist_class = SquashedNormal
        else:
            dist_class = distributions.Normal
        std = log_std.exp()
        dist = dist_class(mu, std)
        return dist

    def predict(self, obs, sample=False):
        dist = self(obs)
        if sample:
            action = dist.sample()
        else:
            action = dist.loc
        action = action.clamp(*self.action_range)
        return action
