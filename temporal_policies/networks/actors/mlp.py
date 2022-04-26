import torch  # type: ignore

from temporal_policies.networks.mlp import MLP, weight_init
from temporal_policies.networks.utils import SquashedNormal


class ContinuousMLPActor(torch.nn.Module):
    def __init__(
        self,
        state_space,
        action_space,
        hidden_layers=[256, 256],
        act=torch.nn.ReLU,
        output_act=torch.nn.Tanh,
        ortho_init=False,
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

    def forward(self, obs):
        return self.mlp(obs)


class DiagonalGaussianMLPActor(torch.nn.Module):
    def __init__(
        self,
        state_space,
        action_space,
        hidden_layers=[256, 256],
        act=torch.nn.ReLU,
        ortho_init=False,
        log_std_bounds=[-5, 2],
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        if log_std_bounds is not None:
            assert log_std_bounds[0] < log_std_bounds[1]
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

    def forward(self, obs):
        mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        if self.log_std_bounds is not None:
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            dist_class = SquashedNormal
        else:
            dist_class = torch.distributions.Normal
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
