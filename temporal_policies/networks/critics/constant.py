import torch

from temporal_policies import networks
from temporal_policies.networks.critics.base import Critic


class ConstantCritic(Critic):
    """Dummy critic that returns constant rewards."""

    def __init__(self, constant: float, dim_states: int):
        """Constructs the oracle critic.

        Args:
            constant: Constant output.
            dim_states: Dimensions of the input state.
        """
        super().__init__()
        self.network = networks.Constant(constant, input_dim=dim_states)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Outputs the constant repeated according to the input batch dimensions.

        Args:
            state: Environment state.
            action: Action.
        """
        return self.network(state)
