import pathlib
import random
from typing import Any, Dict, Optional, OrderedDict, Tuple, Type, Union

import gym
import torch

from temporal_policies import datasets, envs, networks
from temporal_policies.encoders.autoencoder import Autoencoder
from temporal_policies.networks.encoders import beta_tcvae
from temporal_policies.utils import configs
from temporal_policies.utils.typing import AutoencoderBatch


class VAE(Autoencoder):
    """Beta-TCVAE."""

    def __init__(
        self,
        env: envs.Env,
        encoder_class: Union[str, Type[networks.encoders.Encoder]],
        encoder_kwargs: Dict[str, Any],
        decoder_class: Union[str, Type[networks.encoders.Decoder]],
        decoder_kwargs: Dict[str, Any],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        prior_distribution: Union[str, Type[torch.nn.Module]] = beta_tcvae.dist.Normal,
        q_distribution: Union[str, Type[torch.nn.Module]] = beta_tcvae.dist.Normal,
        mutual_information: bool = True,
        tcvae: bool = False,
        stratified_sampling: bool = False,
        beta: float = 1.0,
        kl_reset_steps: int = 10000,
        kl_warmup_steps: int = 5000,
        anneal_beta: bool = True,
        anneal_gamma: bool = False,
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            encoder_class: Dynamics model network class.
            encoder_kwargs: Kwargs for network class.
            state_space: Optional state space.
            action_space: Optional action space.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        super().__init__(
            env=env,
            encoder_class=encoder_class,
            encoder_kwargs=encoder_kwargs,
            decoder_class=decoder_class,
            decoder_kwargs=decoder_kwargs,
            device=device,
        )

        prior_distribution = configs.get_instance(
            prior_distribution, {}, beta_tcvae.dist
        )
        q_distribution = configs.get_instance(q_distribution, {}, beta_tcvae.dist)
        assert isinstance(self.encoder.state_space, gym.spaces.Box)
        self.vae = beta_tcvae.VAE(
            z_dim=self.encoder.state_space.shape[0],
            prior_dist=prior_distribution(),
            q_dist=q_distribution(),
            include_mutinfo=mutual_information,
            tcvae=tcvae,
            mss=stratified_sampling,
            encoder=self.encoder,
            decoder=self.decoder,
        ).to(self.device)
        self.vae.beta = beta

        self.beta = beta
        self.kl_reset_steps = kl_reset_steps
        self.kl_warmup_steps = kl_warmup_steps
        self.anneal_beta = anneal_beta
        self.anneal_gamma = anneal_gamma

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Gets the encoder state dict."""
        return {
            "vae": self.vae.state_dict(),
        }

    def load_state_dict(
        self, state_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict: bool = True
    ):
        """Loads the encoder state dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self.vae.load_state_dict(state_dict["vae"], strict=strict)

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
            "vae": optimizer_class(self.vae.parameters(), **optimizer_kwargs),
        }
        return optimizers

    def train_setup(self, dataset: datasets.StratifiedReplayBuffer) -> None:
        self._dataset_size = len(dataset)

    def compute_loss(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, elbo, reconstruction, latent = self.vae.elbo(
            observation, self._dataset_size
        )
        if torch.isnan(loss).any():
            raise ValueError("NaN spotted in loss.")
        loss = -loss.mean()

        l2_loss = torch.nn.functional.mse_loss(
            reconstruction, observation, reduction="none"
        ).sum(dim=-1)

        metrics = {
            "loss": loss.item(),
            "elbo_loss": -elbo.mean().item(),
            "beta": self.vae.beta,
            "gamma": 1 - self.vae.lamb,
            "l2_loss": l2_loss.mean().item(),
        }
        if not self.vae.training:
            idx_random = random.randrange(observation.shape[0])
            img = observation[idx_random].cpu()
            img_reconstruction = reconstruction[idx_random].cpu()
            metrics["img"] = img.numpy()
            metrics["img_reconstruction"] = img_reconstruction.numpy()
            metrics["emb"] = {
                "mat": latent[idx_random].cpu(),
                "label_img": torch.concat((img, img_reconstruction), dim=-1),
            }

        return loss, metrics

    def anneal_kl(self, step: int) -> None:
        # Cyclical 0 -> 1.
        schedule = min(1.0, (step % self.kl_reset_steps) / self.kl_warmup_steps)

        if self.anneal_beta:
            self.vae.beta = schedule * self.beta  # 0 -> beta.

        if self.anneal_gamma:
            # gamma = 1 - lamb
            self.vae.lamb = max(0.0, 0.95 - schedule)  # gamma = 0.05 -> 1.

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
        self.anneal_kl(step)
        loss, metrics = self.compute_loss(**batch)  # type: ignore

        optimizers["vae"].zero_grad()
        loss.backward()
        optimizers["vae"].step()
        schedulers["vae"].step()

        return metrics

    def to(self, device: Union[str, torch.device]) -> "VAE":
        """Transfers networks to device."""
        super().to(device)
        try:
            self.vae.to(device)
        except AttributeError:
            pass
        return self

    def train_mode(self) -> None:
        """Switches to training mode."""
        super().train_mode()
        self.vae.train()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        super().eval_mode()
        self.vae.eval()
        self.vae.beta = self.beta
        self.vae.lamb = 0.0
