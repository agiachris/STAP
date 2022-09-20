import pathlib
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import numpy as np

from temporal_policies.agents import base as agents
from temporal_policies.agents import rl
from temporal_policies import encoders, envs, networks
from temporal_policies.utils import configs
from temporal_policies.utils.typing import Batch


class SAC(rl.RLAgent):
    """Soft actor critic."""

    def __init__(
        self,
        env: envs.Env,
        actor_class: Union[str, Type[networks.actors.Actor]],
        actor_kwargs: Dict[str, Any],
        critic_class: Union[str, Type[networks.critics.Critic]],
        critic_kwargs: Dict[str, Any],
        encoder: Optional[encoders.Encoder] = None,
        encoder_class: Union[
            str, Type[networks.encoders.Encoder]
        ] = networks.encoders.NormalizeObservation,
        encoder_kwargs: Dict[str, Any] = {},
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        tau: float = 0.005,
        initial_temperature: float = 0.1,
        critic_update_freq: int = 1,
        actor_update_freq: int = 2,
        target_update_freq: int = 2,
    ):
        """Constructs the SAC agent from config parameters.

        Args:
            env: Agent env.
            actor_class: Actor class.
            actor_kwargs: Actor kwargs.
            critic_class: Critic class.
            critic_kwargs: Critic kwargs.
            encoder_class: Encoder class.
            encoder_kwargs: Encoder kwargs.
            checkpoint: Optional policy checkpoint.
            device: Torch device.
            tau: Weighting factor for target update. tau=1.0 replaces the target
                network completely.
            initial_temperature: Initial learning temperature.
            critic_update_freq: Critic update frequency.
            actor_update_freq: Actor update frequency.
            target_update_freq: Target update frequency.
        """
        if encoder is None:
            encoder = encoders.Encoder(env, encoder_class, encoder_kwargs, device)
            target_encoder = encoders.Encoder(
                env, encoder_class, encoder_kwargs, device
            )
            target_encoder.network.load_state_dict(encoder.network.state_dict())
        else:
            target_encoder = encoder

        for param in target_encoder.network.parameters():
            param.requires_grad = False
        target_encoder.eval_mode()

        actor_class = configs.get_class(actor_class, networks)
        actor = actor_class(encoder.state_space, env.action_space, **actor_kwargs)  # type: ignore

        critic_class = configs.get_class(critic_class, networks)
        critic = critic_class(encoder.state_space, env.action_space, **critic_kwargs)  # type: ignore

        target_critic = critic_class(  # type: ignore
            target_encoder.state_space, env.action_space, **critic_kwargs
        )
        target_critic.load_state_dict(critic.state_dict())
        for param in target_critic.parameters():
            param.requires_grad = False
        target_critic.eval()

        self._log_alpha = torch.tensor(
            np.log(initial_temperature), dtype=torch.float, requires_grad=True
        )
        self._target_critic = target_critic
        self._target_encoder = target_encoder

        super().__init__(
            env=env,
            actor=actor,
            critic=critic,
            encoder=encoder,
            checkpoint=checkpoint,
            device=device,
        )

        self.target_entropy = -np.prod(self.action_space.shape)
        self.tau = tau
        self.critic_update_freq = critic_update_freq
        self.actor_update_freq = actor_update_freq
        self.target_update_freq = target_update_freq

    @property
    def log_alpha(self) -> torch.Tensor:
        """Log learning temperature."""
        return self._log_alpha

    @property
    def alpha(self) -> torch.Tensor:
        """Learning temperature."""
        return self.log_alpha.exp()

    @property
    def target_critic(self) -> torch.nn.Module:
        """Target critic."""
        return self._target_critic

    @property
    def target_encoder(self) -> encoders.Encoder:
        """Target encoder."""
        return self._target_encoder

    def to(self, device: Union[str, torch.device]) -> agents.Agent:
        """Transfers networks to device."""
        super().to(device)
        self.target_critic.to(self.device)
        self.target_encoder.to(self.device)
        self.log_alpha.to(self.device)
        return self

    def compute_critic_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        discount: torch.Tensor,
        policy_args: np.ndarray,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the critic loss.

        Args:
            observation: Batch observation.
            action: Batch action.
            reward: Batch reward.
            next_observation: Batch next observation.
            discount: Batch discount.

        Returns:
            2-tuple (critic loss, loss metrics).
        """
        with torch.no_grad():
            dist = self.actor(next_observation)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1)
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob
            target_q = reward + discount * target_v

        q1, q2 = self.critic(observation, action)
        q1_loss = torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
            "target_q": target_q.mean().item(),
        }

        return q_loss, metrics

    def compute_actor_and_alpha_loss(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Computes the actor and learning temperature loss.

        Args:
            observation: Batch observation.

        Returns:
            2-tuple (actor loss, alpha loss, loss metrics).
        """
        obs = observation.detach()  # Detach the encoder so it isn't updated.
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        q1, q2 = self.critic(obs, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_prob.mean().item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

        return actor_loss, alpha_loss, metrics

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
        optimizers = {
            "actor": optimizer_class(self.actor.parameters(), **optimizer_kwargs),
            "critic": optimizer_class(
                self.critic.parameters(),
                # itertools.chain(
                #     self.critic.parameters(), self.encoder.network.parameters()
                # ),
                **optimizer_kwargs,
            ),
            "log_alpha": optimizer_class([self.log_alpha], **optimizer_kwargs),
        }
        return optimizers

    def train_step(
        self,
        step: int,
        batch: Batch,
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
        assert isinstance(batch["observation"], torch.Tensor)
        assert isinstance(batch["next_observation"], torch.Tensor)

        updating_critic = (
            False
            if self.critic_update_freq == 0
            else step % self.critic_update_freq == 0
        )
        updating_actor = (
            False if self.actor_update_freq == 0 else step % self.actor_update_freq == 0
        )
        updating_target = (
            False
            if self.target_update_freq == 0
            else step % self.target_update_freq == 0
        )

        if updating_actor or updating_critic:
            with torch.no_grad():
                batch["observation"] = self.encoder.encode(
                    batch["observation"], batch["policy_args"]
                )
                batch["next_observation"] = self.target_encoder.encode(
                    batch["next_observation"], batch["policy_args"]
                )

        metrics = {}
        if updating_critic:
            q_loss, critic_metrics = self.compute_critic_loss(**batch)  # type: ignore

            optimizers["critic"].zero_grad(set_to_none=True)
            q_loss.backward()
            optimizers["critic"].step()
            schedulers["critic"].step()

            metrics.update(critic_metrics)

        if updating_actor:
            actor_loss, alpha_loss, actor_metrics = self.compute_actor_and_alpha_loss(
                batch["observation"]
            )

            optimizers["actor"].zero_grad(set_to_none=True)
            actor_loss.backward()
            optimizers["actor"].step()
            schedulers["actor"].step()

            optimizers["log_alpha"].zero_grad(set_to_none=True)
            alpha_loss.backward()
            optimizers["log_alpha"].step()
            schedulers["log_alpha"].step()

            metrics.update(actor_metrics)

        if updating_target:
            with torch.no_grad():
                # _update_params(
                #     source=self.encoder.network,
                #     target=self.target_encoder.network,
                #     tau=self.tau,
                # )
                _update_params(
                    source=self.critic, target=self.target_critic, tau=self.tau
                )

        return metrics


def _update_params(
    source: torch.nn.Module, target: torch.nn.Module, tau: float
) -> None:
    """Updates the target parameters towards the source parameters.

    Args:
        source: Source network.
        target: Target network.
        tau: Weight of target update. tau=1.0 sets the target equal to the
            source, and tau=0.0 performs no update.
    """
    for source_params, target_params in zip(source.parameters(), target.parameters()):
        target_params.data.copy_(
            tau * source_params.data + (1 - tau) * target_params.data
        )
