import torch
from torch import nn
from torch.nn import functional as F
import math

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=[256, 256], act=nn.ReLU, output_act=None):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(nn.Linear(last_dim, dim))
            net.append(act())
            last_dim = dim
        net.append(nn.Linear(last_dim, output_dim))
        if not output_act is None:
            net.append(output_act())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class LinearEnsemble(nn.Module):

    def __init__(self, in_features, out_features, bias=True, ensemble_size=3, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((ensemble_size, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError("LinearEnsemble layer does not support inputs with more than 3 dimensions.")
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return 'ensemble_size={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )

class EnsembleMLP(nn.Module):

    def __init__(self, input_dim, output_dim, ensemble_size=3, hidden_layers=[256, 256], act=nn.ReLU, output_act=None):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(LinearEnsemble(last_dim, dim, ensemble_size=ensemble_size))
            net.append(act())
            last_dim = dim
        net.append(LinearEnsemble(last_dim, output_dim, ensemble_size=ensemble_size))
        if not output_act is None:
            net.append(output_act())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
