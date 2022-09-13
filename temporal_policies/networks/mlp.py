from typing import List, Optional, Sequence, Type
import math

import numpy as np
import torch


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int] = [256, 256],
        act: Type[torch.nn.Module] = torch.nn.ReLU,
        output_act: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__()
        net: List[torch.nn.Module] = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(torch.nn.Linear(last_dim, dim))
            net.append(act())
            last_dim = dim
        net.append(torch.nn.Linear(last_dim, output_dim))
        if output_act is not None:
            net.append(output_act())
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


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


class LinearEnsemble(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        ensemble_size=3,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = torch.nn.Parameter(
            torch.empty((ensemble_size, in_features, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((ensemble_size, 1, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError(
                "LinearEnsemble layer does not support inputs with more than 3 dimensions."
            )
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "ensemble_size={}, in_features={}, out_features={}, bias={}".format(
            self.ensemble_size,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


class EnsembleMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        ensemble_size=3,
        hidden_layers=[256, 256],
        act=torch.nn.ReLU,
        output_act=None,
    ):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(LinearEnsemble(last_dim, dim, ensemble_size=ensemble_size))
            net.append(act())
            last_dim = dim
        net.append(LinearEnsemble(last_dim, output_dim, ensemble_size=ensemble_size))
        if output_act is not None:
            net.append(output_act())
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
