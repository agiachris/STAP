import pathlib
from typing import Any, Dict, Optional, Tuple, Type, Union, List

import torch
import numpy as np
from copy import deepcopy

from temporal_policies.agents import base as agents
from temporal_policies.agents import rl
from temporal_policies import encoders, envs, networks
from temporal_policies.utils import configs
from temporal_policies.utils.typing import Batch


def get_bce_weight(target_q: torch.Tensor, logistics_weight: torch.Tensor):
    """Get logistics regression weight."""
    weight = torch.zeros_like(target_q, dtype=torch.float32).to(target_q.device)
    weight[target_q == 0] = logistics_weight[0]
    weight[target_q == 1] = logistics_weight[1]
    return weight
    

class SAC(rl.RLAgent):
    """Soft actor critic."""

    def __init__(
        self,
        env: envs.Env,
        actor_class: Union[str, Type[networks.actors.Actor]],
        actor_kwargs: Dict[str, Any],
        critic_class: Union[str, Type[networks.critics.Critic]],
        critic_kwargs: Dict[str, Any],
        encoder_class: Union[
            str, Type[networks.encoders.Encoder]
        ] = networks.encoders.NormalizeObservation,
        encoder_kwargs: Dict[str, Any] = {},
        actor: Optional[networks.actors.Actor] = None,
        critic: Optional[networks.critics.Critic] = None,
        encoder: Optional[encoders.Encoder] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        tau: float = 0.005,
        initial_temperature: float = 0.1,
        critic_update_freq: int = 1,
        actor_update_freq: int = 2,
        target_update_freq: int = 2,
        use_bce: bool = False,
        bce_weight: Optional[List[float]] = None
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
            actor: Custom actor.
            critic: Custom critic.
            encoder: Custom encoder.
            checkpoint: Optional policy checkpoint.
            device: Torch device.
            tau: Weighting factor for target update. tau=1.0 replaces the target
                network completely.
            initial_temperature: Initial learning temperature.
            critic_update_freq: Critic update frequency.
            actor_update_freq: Actor update frequency.
            target_update_freq: Target update frequency.
            use_bce: Logstics regression loss instead of MSE.
            bce_weight: Logistics regression classification weight.
        """
        agent_kwargs = {
            "actor": deepcopy(actor_kwargs),
            "critic": deepcopy(critic_kwargs),
            "encoder": deepcopy(encoder_kwargs),
        }
        for kwargs in agent_kwargs.values():
            for key in ["act", "output_act"]:
                if kwargs.get(key, False):
                    kwargs[key] = configs.get_class(kwargs[key], torch.nn)

        if encoder is None:
            encoder = encoders.Encoder(env, encoder_class, agent_kwargs["encoder"], device)
            target_encoder = encoders.Encoder(
                env, encoder_class, agent_kwargs["encoder"], device
            )
            target_encoder.network.load_state_dict(encoder.network.state_dict())
        else:
            target_encoder = encoder

        for param in target_encoder.network.parameters():
            param.requires_grad = False
        target_encoder.eval_mode()

        actor_class = configs.get_class(actor_class, networks)
        if actor is None:
            actor = actor_class(encoder.state_space, env.action_space, **agent_kwargs["actor"])  # type: ignore
        
        critic_class = configs.get_class(critic_class, networks)
        if critic is None:
            critic = critic_class(encoder.state_space, env.action_space, **agent_kwargs["critic"])  # type: ignore

        target_critic = critic_class(  # type: ignore
            target_encoder.state_space, env.action_space, **agent_kwargs["critic"]
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

        if isinstance(bce_weight, list):
            if len(bce_weight) != 2 or any(w <= 0 for w in bce_weight):
                raise ValueError("Require non-negative weight for positive and negative classes.")
            self.bce_weight = torch.tensor(bce_weight).float().to(self.device) / sum(bce_weight)
        else:
            self.bce_weight = None

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
        self.use_bce = use_bce

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
        if isinstance(self.bce_weight, torch.Tensor):
            self.bce_weight.to(self.device)
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
            target_q = self.target_critic(next_observation, next_action)
            target_q = torch.min(torch.stack(target_q), axis=0).values
            target_v = target_q - self.alpha.detach() * log_prob
            target_q: torch.Tensor = reward + discount * target_v

        qs = self.critic(observation, action)
        if self.use_bce:
            if len(target_q) != (target_q == 0.0).sum() + (target_q == 1.0).sum():
                raise ValueError("Logistics regression requires [0, 1] targets.")
            weight = get_bce_weight(target_q, self.bce_weight) if isinstance(self.bce_weight, torch.Tensor) else None
            q_losses = [torch.nn.functional.binary_cross_entropy(q, target_q, weight=weight) for q in qs]
        else:
            q_losses = [torch.nn.functional.mse_loss(q, target_q) for q in qs]
        q_loss = sum(q_losses) / len(q_losses)

        metrics = {f"q{i}_loss": q.item() for i, q in enumerate(q_losses)}
        metrics.update({
            "q_loss": q_loss.item(),
            "target_q": target_q.mean().item(),
        })

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
        q = self.critic(obs, action)
        q = torch.min(torch.stack(q), axis=0).values
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
        keys = ["actor", "critic", "log_alpha"]    
        if (
            any(k not in optimizer_kwargs for k in keys)
            and any(k in optimizer_kwargs for k in keys)
            ):
            raise ValueError(f"Must supply general optimizer_kwargs or optimizer_kwargs for each of {keys}")

        optimizers = {}
        for key in keys:
            try:
                kwargs = optimizer_kwargs[key]
            except KeyError:
                kwargs = optimizer_kwargs
            
            if not hasattr(self, key):
                raise ValueError(f"SAC does not have attribute self.{key}.")
            learnable = getattr(self, key)

            if isinstance(learnable, torch.nn.Module):
                parameters = learnable.parameters()
            elif isinstance(learnable, torch.Tensor):
                parameters = [learnable]
            else:
                raise ValueError(f"Optimization not supported for {learnable}.")

            optimizers[key] = optimizer_class(parameters, **kwargs)
            
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
                _update_params(
                    source=self.critic, target=self.target_critic, tau=self.tau
                )

        return metrics

    def validation_step(
        self,
        batch: Batch,
    ) -> Dict[str, Any]:
        """Performs a single validation step.

        Args:
            batch: Validation batch.

        Returns:
            Dict of loggable validation metrics.
        """
        evaluating_critic = self.critic_update_freq > 0
        evaluating_actor = self.actor_update_freq > 0

        if evaluating_critic or evaluating_actor:
            with torch.no_grad():
                batch["observation"] = self.encoder.encode(
                    batch["observation"], batch["policy_args"]
                )
                batch["next_observation"] = self.target_encoder.encode(
                    batch["next_observation"], batch["policy_args"]
                )

        metrics = {}
        if self.critic_update_freq > 0:
            _, critic_metrics = self.compute_critic_loss(**batch)  # type: ignore
            metrics.update(critic_metrics)

        if self.actor_update_freq > 0:
            _, _, actor_metrics = self.compute_actor_and_alpha_loss(
                batch["observation"]
            )
            metrics.update(actor_metrics)

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