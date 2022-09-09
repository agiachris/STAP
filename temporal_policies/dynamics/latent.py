import pathlib
from typing import Any, Dict, Optional, OrderedDict, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch

from temporal_policies import agents, networks
from temporal_policies.dynamics.base import Dynamics
from temporal_policies.utils import configs
from temporal_policies.utils.typing import DynamicsBatch, Model, Scalar


class LatentDynamics(Dynamics, Model[DynamicsBatch]):
    """Base dynamics class."""

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[networks.dynamics.Dynamics]],
        network_kwargs: Dict[str, Any],
        state_space: Optional[gym.spaces.Box] = None,
        action_space: Optional[gym.spaces.Box] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Dynamics model network class.
            network_kwargs: Kwargs for network class.
            state_space: Optional state space.
            action_space: Optional action space.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        network_class = configs.get_class(network_class, networks)
        self._network = network_class(**network_kwargs)

        super().__init__(
            policies=policies,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        if checkpoint is not None:
            self.load(checkpoint, strict=True)

    @property
    def network(self) -> networks.dynamics.Dynamics:
        """Dynamics model network."""
        return self._network

    def load_state_dict(
        self, state_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict: bool = True
    ):
        """Loads the dynamics state dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self.network.load_state_dict(state_dict["dynamics"], strict=strict)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Gets the dynamics state dict."""
        return {
            "dynamics": self.network.state_dict(),
        }

    def create_optimizers(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.optim.Optimizer]:
        """Creates the optimizers for training.

        This method is called by the Trainer class.

        Args:
            optimizer_class: Optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
        Returns:
            Dict of optimizers.
        """
        optimizers = {
            "dynamics": optimizer_class(self.network.parameters(), **optimizer_kwargs)
        }
        return optimizers

    def to(self, device: Union[str, torch.device]) -> "LatentDynamics":
        """Transfers networks to device."""
        super().to(device)
        self.network.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        self.network.eval()

    def plan_mode(self) -> None:
        """Switches to plan mode."""
        self.eval_mode()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Any]],
    ) -> torch.Tensor:
        """Predicts the next latent state given the current latent state and
        action.

        Args:
            state: Current latent state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next latent state.
        """
        dz = self.network(state, idx_policy, action)
        return state + dz

    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        idx_policy: torch.Tensor,
        policy_args: np.ndarray,
    ) -> Tuple[torch.Tensor, Dict[str, Union[Scalar, np.ndarray]]]:
        """Computes the L2 loss between the predicted next latent and the latent
        encoded from the given next observation.

        Args:
            observation: Common observation across all policies.
            action: Policy parameters.
            next_observation: Next observation.
            idx_policy: Index of executed policy.

        Returns:
            L2 loss.
        """
        # Predict next latent state.
        # [B, 3, H, W], [B] => [B, Z].
        latent = self.encode(observation, idx_policy, policy_args)

        # [B, Z], [B], [B, A] => [B, Z].
        next_latent_pred = self.forward(latent, action, idx_policy, policy_args)

        # Encode next latent state.
        # [B, 3, H, W], [B] => [B, Z].
        next_latent = self.encode(next_observation, idx_policy, policy_args)

        # Compute L2 loss.
        # [B, Z], [B, Z] => [1].
        l2_loss = torch.nn.functional.mse_loss(next_latent_pred, next_latent)

        metrics: Dict[str, Union[Scalar, np.ndarray]] = {
            "l2_loss": l2_loss.item(),
        }

        # Compute per-policy L2 losses.
        # [B, Z], [B, Z] => [B].
        l2_losses = torch.nn.functional.mse_loss(
            next_latent_pred, next_latent, reduction="none"
        ).sum(dim=-1)

        # if isinstance(idx_policy, int):
        #     # [B] => [B, P].
        #     policy_l2_losses = torch.zeros(
        #         *l2_losses.shape,
        #         len(self.policies),
        #         dtype=l2_losses.dtype,
        #         device=self.device,
        #     )
        #     policy_l2_losses[..., idx_policy] = l2_losses
        # else:
        # [B], [P] => [B, P].
        idx_policies = idx_policy.unsqueeze(-1) == torch.arange(
            len(self.policies), device=self.device
        )

        # [B] => [B, P].
        l2_losses = l2_losses.unsqueeze(-1).tile((len(self.policies),))

        # [B] => [B, P].
        policy_l2_losses = l2_losses * idx_policies

        # [B, P], [B, P] => [P].
        batch_dims = list(range(len(l2_losses.shape) - 1))
        policy_l2_losses = policy_l2_losses.sum(dim=batch_dims) / idx_policies.sum(
            dim=batch_dims
        )

        for i_policy, policy_l2_loss in enumerate(
            policy_l2_losses.detach().cpu().numpy()
        ):
            metrics[f"l2_loss_policy_{i_policy}"] = policy_l2_loss

        return l2_loss, metrics

    def train_step(
        self,
        step: int,
        batch: DynamicsBatch,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    ) -> Dict[str, Union[Scalar, np.ndarray]]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.
            optimizers: Optimizers created in `LatentDynamics.create_optimizers()`.
            schedulers: Schedulers with the same keys as `optimizers`.

        Returns:
            Dict of training metrics for logging.
        """
        loss, metrics = self.compute_loss(**batch)  # type: ignore

        optimizers["dynamics"].zero_grad()
        loss.backward()
        optimizers["dynamics"].step()
        schedulers["dynamics"].step()

        return metrics
