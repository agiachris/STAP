import torch
from torch import nn
from torch.nn import functional as F
import gym

from .base import ActorCriticPolicy
from .common import MLP

class ContinuousMLPQCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, num_q_fns=2):
        super().__init__()
        self.qs = nn.ModuleList([
            MLP(observation_space.shape[0] + action_space.shape[0], 1, hidden_layers=hidden_layers, act=act)
         for _ in range(num_q_fns)])

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]

class ContinuousMLPPolicy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU, output_act=nn.Tanh):
        super().__init__()
        self.mlp = MLP(observation_space.shape[0], action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=output_act)

    def forward(self, obs):
        return self.mlp(obs)

class ContinuousActorCriticMLP(nn.Module, ActorCriticPolicy):
    
    def __init__(self, observation_space, action_space, 
                       actor_hidden_layers=[256, 256],
                       critic_hidden_layers=[256, 256],
                       act=nn.ReLU,
                       num_q_fns=2,
                       actor_output_act=nn.Tanh):
        super().__init__()
        self._actor = ContinuousMLPPolicy(observation_space, action_space, 
                                          hidden_layers=actor_hidden_layers,
                                          act=act, output_act=actor_output_act)
        self._critic = ContinuousMLPQCritic(observation_space, action_space,
                                                hidden_layers=critic_hidden_layers,
                                                act=act, num_q_fns=num_q_fns)

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic

    def predict(self, obs):
        return self._actor(obs)
