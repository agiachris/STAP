import torch
from torch import nn
from torch.nn import functional as F
import gym
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class DRQV2Encoder(nn.Module):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        if len(observation_space.shape) == 4:
            s, c, h, w = observation_space.shape
            channels = s * c
        elif len(observation_space.shape) == 3:
            c, h, w = observation_space.shape
            channels = c
        else:
            raise ValueError("Invalid observation space for DRQV2 Image encoder.")
        assert h == w == 84, "Incorrect spatial dimensions for DRQV2 Encoder"
        # For the future modules
        self.output_space = gym.spaces.Box(
            shape=(32 * 35 * 35,), low=-np.inf, high=np.inf, dtype=np.float32
        )
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.apply(weight_init)

    def forward(self, obs):
        if len(obs.shape) == 5:
            b, s, c, h, w = obs.shape
            obs = obs.view(b, s * c, h, w)
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class DRQV2Critic(nn.Module):
    def __init__(
        self, observation_space, action_space, feature_dim=50, hidden_dim=1024
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        return self.Q1(h_action).squeeze(-1), self.Q2(h_action).squeeze(-1)


class DRQV2Actor(nn.Module):
    def __init__(
        self, observation_space, action_space, feature_dim=50, hidden_dim=1024
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
        )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        return mu
