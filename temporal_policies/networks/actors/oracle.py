import numpy as np
import torch

from temporal_policies import envs
from temporal_policies.networks.actors.base import Actor
from temporal_policies.utils import tensors


class OracleActor(Actor):
    """Wrapper actor that converts ground truth states to observations before
    passing to the child actor."""

    def __init__(self, env: envs.Env, policy):
        """Constructs the oracle actor.

        Args:
            env: Env for simulation.
            policy: Child actor policy.
        """
        super().__init__()
        self.env = env
        self.encoder = policy.encoder
        self.actor = policy.actor

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def _get_observation(self, state: np.ndarray) -> np.ndarray:
        """Gets the policy observation from the environment."""
        self.env.set_state(state)
        return self.env.get_observation()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs the predicted distribution from the child policy.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        observation = self._get_observation(state)
        policy_state = self.encoder(observation)
        return self.actor(policy_state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs the prediction from the child policy.

        Args:
            state: Environment state.

        Returns:
            Action.
        """
        observation = self._get_observation(state)
        policy_state = self.encoder(observation)
        return self.actor.predict(policy_state)
