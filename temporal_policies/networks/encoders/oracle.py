import numpy as np  # type: ignore

from temporal_policies import envs
from temporal_policies.networks.encoders.base import Encoder
from temporal_policies.utils import tensors


class OracleEncoder(Encoder):
    """Dummy encoder that returns the ground truth environment state.

    For use with the OracleAgent.
    """

    def __init__(self, env: envs.Env):
        """Constructs the oracle encoder.

        Args:
            env: Gym environment.
        """
        super().__init__(env, env.state_space)

        @tensors.vmap(dims=len(self.env.observation_space.shape))
        def forward(observation: np.ndarray) -> np.ndarray:
            if (observation != self.env.get_observation()).any():
                raise ValueError("Observation does not match the current env state")
            return self.env.get_state()

        self._forward = forward

    @tensors.torch_wrap
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Returns the current environment state.

        WARNING: This ignores the input observation and instead returns the
        environment's current ground truth state. Be careful that the state
        matches the observation as expected.
        """
        return self._forward(observation)
