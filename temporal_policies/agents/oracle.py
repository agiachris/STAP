from typing import Optional, Sequence, Union

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import envs, networks


class OracleAgent(agents.Agent):
    """Dummy agent that features an oracle critic."""

    def __init__(
        self,
        env: envs.Env,
        action: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        device: str = "auto",
    ):
        """Constructs the oracle agent.

        If action is specified, the agent's actor will always output it.
        Otherwise, the actor will output random actions.

        Args:
            env: Policy env.
            action: Optional constant action.
            device: Torch device.
        """
        dim_states = len(env.state_space.shape)
        if action is not None:
            actor = networks.Constant(action, input_dim=dim_states)
        else:
            actor = networks.Random(
                min=env.action_space.low,
                max=env.action_space.high,
                input_dim=dim_states,
            )

        super().__init__(
            state_space=env.state_space,
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor=actor,
            critic=networks.critics.OracleCritic(env),
            encoder=networks.encoders.OracleEncoder(env),
            device=device,
        )

        self._env = env

    @property
    def env(self) -> envs.Env:
        """Last generated env."""
        return self._env

    @env.setter
    def env(self, env: envs.Env) -> None:
        """Sets the last generated env."""
        self._env = env
        self.critic.env = env
        self.encoder.env = env
