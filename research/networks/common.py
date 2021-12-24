import torch
from torch import nn
from torch.nn import functional as F

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
