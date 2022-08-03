import pathlib
from typing import Dict, Optional, OrderedDict, Union

import torch

from temporal_policies import encoders, envs, networks
from temporal_policies.agents.base import Agent
from temporal_policies.utils.typing import Batch, Model


class RLAgent(Agent, Model[Batch]):
    """RL agent base class."""

    def __init__(
        self,
        env: envs.Env,
        actor: networks.actors.Actor,
        critic: networks.critics.Critic,
        encoder: encoders.Encoder,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Sets up the agent and loads from checkpoint if available.

        Args:
            env: Agent env.
            actor: Actor network.
            critic: Critic network.
            encoder: Encoder network.
            checkpoint: Policy checkpoint.
            device: Torch device.
        """
        super().__init__(
            state_space=encoder.state_space,
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor=actor,
            critic=critic,
            encoder=encoder,
            device=device,
        )

        self._env = env

        if checkpoint is not None:
            self.load(checkpoint, strict=True)

    @property
    def env(self) -> envs.Env:
        """Agent environment."""
        return self._env

    def load_state_dict(
        self, state_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict: bool = True
    ) -> None:
        """Loads the agent state dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.encoder.network.load_state_dict(state_dict["encoder"], strict=strict)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Gets the agent state dicts."""
        return {
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "encoder": self.encoder.network.state_dict(),
        }
