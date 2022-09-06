import torch

from temporal_policies import envs
from temporal_policies.networks.actors.base import Actor
from temporal_policies.networks.encoders.oracle import OracleDecoder


class OracleActor(Actor):
    """Wrapper actor that converts ground truth states to observations before
    passing to the child actor."""

    def __init__(self, env: envs.Env, policy):
        """Constructs the oracle actor.

        Args:
            env: Env for simulation.
            policy: Child actor policy (temporal_policies.agents.base.Agent).
        """
        super().__init__()
        self.env = env
        self.encoder = policy.encoder
        self.actor = policy.actor

        self._oracle_decoder = OracleDecoder(self.env)

    def reset_cache(self) -> None:
        self._oracle_decoder.reset_cache()

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """Outputs the predicted distribution from the child policy.

        Args:
            state: Environment state.

        Returns:
            Action distribution.
        """
        observation = self._oracle_decoder(state)
        policy_args = self.env.get_primitive().get_policy_args()
        policy_state = self.encoder.encode(observation, policy_args)

        return self.actor(policy_state)

    def predict(self, state: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Outputs the prediction from the child policy.

        Args:
            state: Environment state.
            sample: Whether to sample from the distribution or return the mode.

        Returns:
            Action.
        """
        observation = self._oracle_decoder(state)
        policy_args = self.env.get_primitive().get_policy_args()
        policy_state = self.encoder.encode(observation, policy_args)

        return self.actor.predict(policy_state, sample=sample)
