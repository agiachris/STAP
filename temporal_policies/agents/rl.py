import abc
import pathlib
from typing import Any, Dict, Optional, Type, Union

import torch  # type: ignore

from temporal_policies import envs, networks
from temporal_policies.agents.base import Agent


class RLAgent(Agent, abc.ABC):
    """RL agent base class."""

    def __init__(
        self,
        env: envs.Env,
        actor: networks.actors.Actor,
        critic: networks.critics.Critic,
        encoder: networks.encoders.Encoder,
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
            state_space=env.observation_space,
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
        self, state_dict: Dict[str, Dict[str, torch.Tensor]], strict: bool = True
    ) -> None:
        """Loads the model state from the state_dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.encoder.load_state_dict(state_dict["encoder"], strict=strict)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Gets the model state dicts."""
        return {
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "encoder": self.encoder.state_dict(),
        }

    def load(self, checkpoint: Union[str, pathlib.Path], strict: bool = True) -> None:
        """Loads the model from the given checkpoint.

        Args:
            checkpoint: Checkpoint path.
            strict: Make sure the state dict keys match.
        """
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, path: Union[str, pathlib.Path], name: str) -> None:
        """Saves a checkpoint of the model.

        Args:
            path: Directory of checkpoint.
            name: Name of checkpoint (saved as `path/name.pt`).
        """
        torch.save(self.state_dict(), pathlib.Path(path) / f"{name}.pt")

    @abc.abstractmethod
    def create_optimizers(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.optim.Optimizer]:
        """Sets up the agent optimizers.

        This function is called by the agent trainer, since the optimizer class
        is only required during training.

        Args:
            optimizer_class: Optimizer class.
            optimizer_kwargs: Optimizer kwargs.

        Returns:
            Dict of optimizers for all trainable networks.
        """
        pass

    @abc.abstractmethod
    def train_step(
        self,
        step: int,
        batch: Dict[str, Any],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    ) -> Dict[str, Any]:
        """Performs a single training step.

        Args:
            step: Step index.
            batch: Training batch.
            optimizers: Optimizers created in `RLAgent.create_optimizers()`.
            schedulers: Schedulers with the same keys as `optimizers`.

        Returns:
            Dict of loggable training metrics.
        """
        pass
