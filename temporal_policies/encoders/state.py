import pathlib
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch

from temporal_policies import envs, networks
from temporal_policies.encoders.base import Encoder
from temporal_policies.utils.typing import StateEncoderBatch, Model, Scalar


class StateEncoder(Encoder, Model[StateEncoderBatch]):
    """Vanilla autoencoder."""

    def __init__(
        self,
        env: envs.Env,
        encoder_class: Union[str, Type[networks.encoders.Encoder]],
        encoder_kwargs: Dict[str, Any],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the autoencoder network.

        Args:
            env: Encoder env.
            encoder_class: Encoder network class.
            encoder_kwargs: Kwargs for encoder network class.
            decoder_class: decoder network class.
            decoder_kwargs: Kwargs for decoder network class.
            checkpoint: Autoencoder checkpoint.
            device: Torch device.
        """
        super().__init__(
            env=env,
            network_class=encoder_class,
            network_kwargs=encoder_kwargs,
            device=device,
        )

        if checkpoint is not None:
            self.load(checkpoint, strict=True)

    def compute_loss(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        policy_args: np.ndarray,
    ) -> Tuple[torch.Tensor, Dict[str, Union[Scalar, np.ndarray]]]:
        state_prediction = self.network.predict(observation, policy_args)
        loss = torch.nn.functional.mse_loss(state_prediction, state)
        metrics: Dict[str, Union[Scalar, np.ndarray]] = {
            "loss": loss.item(),
        }

        return loss, metrics

    def train_step(
        self,
        step: int,
        batch: StateEncoderBatch,
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
        assert isinstance(batch["observation"], torch.Tensor)
        loss, metrics = self.compute_loss(**batch)  # type: ignore

        optimizers["encoder"].zero_grad()
        loss.backward()
        optimizers["encoder"].step()

        return metrics

    def train_mode(self) -> None:
        """Switches to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        self.network.eval()
