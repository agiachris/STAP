from typing import Any, Optional, Union

import torch
import numpy as np

from temporal_policies import envs
from temporal_policies.networks.encoders import Encoder


class IdentityEncoder(Encoder):
    """Dummy encoder."""

    def __init__(self, env: envs.Env):
        super().__init__(env, env.observation_space)

    def forward(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs
    ) -> torch.Tensor:
        """Returns the original observation."""
        return observation
