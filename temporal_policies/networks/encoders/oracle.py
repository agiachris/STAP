import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import envs
from temporal_policies.utils import tensors


class OracleEncoder(torch.nn.Module):
    """Dummy encoder that returns the ground truth environment state.

    For use with the OracleAgent.
    """

    def __init__(self, env: envs.Env):
        """Constructs the constant network.

        Args:
            constant: Constant output.
            input_dim: Dimensions of the network's first input.
        """
        super().__init__()
        self.env = env

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Returns the current environment state.

        WARNING: This ignores the input observation and instead returns the
        environment's current ground truth state. Be careful that the state
        matches the observation as expected.
        """
        if (observation != self.env.get_observation()).any():
            raise ValueError("Observation does not match the current env state")
        return self.env.get_state()
