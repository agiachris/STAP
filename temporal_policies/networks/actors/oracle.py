import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import agents, envs
from temporal_policies.utils import tensors


class OracleActor(torch.nn.Module):
    """Dummy actor that returns ground truth rewards from simulation."""

    def __init__(self, env: envs.Env, policy: agents.Agent):
        """Constructs the oracle actor.

        Args:
            env: Env for simulation.
            agent: Actor policy for oracle.
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
        """Outputs the prediction from the child policy.

        Args:
            state: Environment state.
            action: Action.
        """
        observation = self._get_observation(state)
        policy_state = self.encoder(observation)
        return self.actor(policy_state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs the reward from the given state and action.

        Args:
            state: Environment state.
            action: Action.
        """
        observation = self._get_observation(state)
        policy_state = self.encoder(observation)
        return self.actor.predict(policy_state)
