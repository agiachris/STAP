import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
import gym
import math

from .base import ActorCriticPolicy
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

# class SquashedNormal(distributions.TransformedDistribution):

#     def __init__(self, loc, scale):
#         self._loc = loc
#         self.scale = scale
#         self.base_dist = distributions.Normal(loc, scale)
#         transforms = [distributions.transforms.TanhTransform(cache_size=1)]
#         super().__init__(self.base_dist, transforms)

#     @property
#     def loc(self):
#         loc = self._loc
#         for transform in self.transforms:
#             loc = transform(loc)
#         return loc

# class DiagonalGaussianMLPActor(nn.Module):

#     def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, ortho_init=False, log_std_bounds=[-5, 2]):
#         super().__init__()
#         self.log_std_bounds = log_std_bounds
#         if log_std_bounds is not None:
#             assert log_std_bounds[0] < log_std_bounds[1]
#         self.mlp = MLP(observation_space.shape[0], 2*action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=None)
#         if ortho_init:
#             self.apply(weight_init)
#         self.action_range = [float(action_space.low.min()), float(action_space.high.max())]
#         print("AC RANGE", self.action_range)
        
#     def forward(self, obs):
#         mu, log_std = self.mlp(obs).chunk(2, dim=-1)
#         if self.log_std_bounds is not None:
#             log_std = torch.tanh(log_std)
#             log_std_min, log_std_max = self.log_std_bounds
#             log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
#         std = log_std.exp()
#         dist = SquashedNormal(mu, std)
#         return dist

#     def predict(self, obs, sample=False):
#         dist = self(obs)
#         if sample:
#             action = dist.sample()
#         else:
#             action = dist.loc
#         action = action.clamp(*self.action_range)
#         return action

from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

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
        
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist

    def predict(self, obs, sample=False):
        dist = self(obs)
        if sample:
            action = dist.sample()
        else:
            action = dist.mean
        action = action.clamp(*self.action_range)
        return action