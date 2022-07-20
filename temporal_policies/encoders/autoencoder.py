import pathlib
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch

from temporal_policies import envs, networks
from temporal_policies.encoders.base import Encoder
from temporal_policies.utils import configs
from temporal_policies.utils.typing import AutoencoderBatch, Model


class Autoencoder(Encoder, Model[AutoencoderBatch]):
    """Vanilla autoencoder."""

    def __init__(
        self,
        env: envs.Env,
        encoder_class: Union[str, Type[networks.encoders.Encoder]],
        encoder_kwargs: Dict[str, Any],
        decoder_class: Union[str, Type[networks.encoders.Decoder]],
        decoder_kwargs: Dict[str, Any],
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
        decoder_class = configs.get_class(decoder_class, networks)
        self._decoder = decoder_class(env, **decoder_kwargs)

        super().__init__(
            env=env,
            network_class=encoder_class,
            network_kwargs=encoder_kwargs,
            device=device,
        )

        if checkpoint is not None:
            self.load(checkpoint, strict=True)

    @property
    def encoder(self) -> torch.nn.Module:
        return self.network

    @property
    def decoder(self) -> torch.nn.Module:
        return self._decoder

    def compute_loss(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def train_step(
        self,
        step: int,
        batch: AutoencoderBatch,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    ) -> Dict[str, float]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.
            optimizers: Optimizers created in `LatentDynamics.create_optimizers()`.
            schedulers: Schedulers with the same keys as `optimizers`.

        Returns:
            Dict of training metrics for logging.
        """
        raise NotImplementedError

    def to(self, device: Union[str, torch.device]) -> "Autoencoder":
        """Transfers networks to device."""
        super().to(device)
        self.decoder.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches to training mode."""
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        self.encoder.eval()
        self.decoder.eval()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
