import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import envs
from temporal_policies.utils import tensors


class OracleCritic(torch.nn.Module):
    """Dummy critic that returns ground truth rewards from simulation."""

    def __init__(self, env: envs.Env):
        """Constructs the oracle critic.

        Args:
            env: Env for simulation.
        """
        super().__init__()
        self.env = env

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Outputs the reward from the given state and action.

        Args:
            state: Environment state.
            action: Action.
        """
        self.env.set_state(state)
        _, reward, _, _ = self.env.step(action)
        return np.array(reward, dtype=np.float32)

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Outputs the reward from the given state and action.

        Args:
            state: Environment state.
            action: Action.
        """
        return self.forward(state, action)
