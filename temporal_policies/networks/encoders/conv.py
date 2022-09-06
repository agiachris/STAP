from typing import Any, List, Optional, Sequence, Type, Union

import gym
import numpy as np
import torch

from temporal_policies import envs
from temporal_policies.networks.encoders.base import Encoder, Decoder


class ConvEncoder(Encoder):
    def __init__(
        self,
        env: envs.Env,
        latent_dim: int,
        hidden_channels: Sequence[int],
        nonlinearity: Type[torch.nn.Module] = torch.nn.ReLU,
        distribution_parameters: int = 2,
    ):
        state_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(latent_dim,),
            dtype=np.float32,
        )
        super().__init__(env, state_space)

        dim_image = np.array(env.observation_space.shape[:2])
        in_channels = env.observation_space.shape[-1]

        layers = []
        # for out_channels in hidden_channels:
        #     conv = torch.nn.Conv2d(
        #         in_channels, out_channels, kernel_size=4, stride=2, padding=1
        #     )
        #     batch_norm = torch.nn.BatchNorm2d(out_channels)
        #     layer = torch.nn.Sequential(conv, batch_norm, nonlinearity())
        #
        #     layers.append(layer)
        #     in_channels = out_channels
        #     dim_image //= 2
        for out_channels in hidden_channels[:-1]:
            conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
            # batch_norm = torch.nn.BatchNorm2d(out_channels)
            # layer = torch.nn.Sequential(conv, batch_norm, nonlinearity())
            layer = torch.nn.Sequential(conv, nonlinearity())

            layers.append(layer)
            in_channels = out_channels
            dim_image //= 2

        assert (dim_image == 4).all()
        layers += [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, hidden_channels[-1], kernel_size=4),
                # torch.nn.BatchNorm2d(hidden_channels[-1]),
                nonlinearity(),
            ),
        ]
        self.encoder = torch.nn.Sequential(*layers)

        # self.fc = torch.nn.Sequential(
        #     torch.nn.Flatten(start_dim=-3),
        #     torch.nn.Linear(
        #         dim_image.prod() * out_channels, latent_dim * distribution_parameters
        #     ),
        # )
        self.fc = torch.nn.Conv2d(
            hidden_channels[-1], latent_dim * distribution_parameters, kernel_size=1
        )

        self._latent_dim = latent_dim

    def forward(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs
    ) -> torch.Tensor:
        """Encodes the observation to the policy latent state.

        Args:
            observation: Environment observation.

        Returns:
            Encoded policy state.
        """

        # [B, 3, 64, 64] => [B, 256, 2, 2].
        features = self.encoder(observation)

        # [B, 256, 2, 2] => [B, latent_dim * distribution_parameters].
        latent = self.fc(features)
        latent = latent.squeeze(-1).squeeze(-1)

        return latent

    def predict(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs
    ) -> torch.Tensor:
        # Make sure input has one batch dimension.
        if observation.dim() < 4:
            squeeze = True
            observation = observation.unsqueeze(0)
        else:
            squeeze = False

        # [B, H, W, 3] => [B, 3, H, W].
        if observation.shape[-1] == 3:
            observation = torch.moveaxis(observation, -1, -3)

        # [B, 3, H, W] => [B, latent_dim * 2].
        latent = self.forward(observation, policy_args)

        # [B, latent_dim * 2] => [B, latent_dim, 2].
        latent = latent.view(*observation.shape[:-3], self._latent_dim, -1)

        # [B, latent_dim, 2] => [B, latent_dim].
        latent = latent.select(-1, 0)

        # Restore original input dimensions.
        if squeeze:
            latent = latent.squeeze(0)

        return latent


class ConvDecoder(Decoder):
    def __init__(
        self,
        env: envs.Env,
        latent_dim: int,
        hidden_channels: Sequence[int],
        nonlinearity: Type[torch.nn.Module] = torch.nn.ReLU,
        distribution_parameters: int = 2,
    ):
        super().__init__(env)

        # dim_image = np.array(env.observation_space.shape[:2]) // (
        #     2 ** len(hidden_channels)
        # )

        in_channels = hidden_channels[0]
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(latent_dim, dim_image.prod() * in_channels),
        #     torch.nn.Unflatten(-1, (in_channels, *dim_image.tolist())),
        # )
        self.fc = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dim, in_channels, 1, 1, 0),
            # torch.nn.BatchNorm2d(in_channels),
            nonlinearity(),
        )

        layers: List[torch.nn.Module] = []
        layers.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, hidden_channels[1], 4, 1, 0),
                # torch.nn.BatchNorm2d(hidden_channels[1]),
                nonlinearity(),
            )
        )
        in_channels = hidden_channels[1]
        # for out_channels in hidden_channels[1:]:
        for out_channels in hidden_channels[2:]:
            conv = torch.nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
            # batch_norm = torch.nn.BatchNorm2d(out_channels)
            # layer = torch.nn.Sequential(conv, batch_norm, nonlinearity())
            layer = torch.nn.Sequential(conv, nonlinearity())

            layers.append(layer)
            in_channels = out_channels

        out_channels = env.observation_space.shape[-1]
        # layer = torch.nn.Sequential(
        #     # torch.nn.ConvTranspose2d(
        #     #     in_channels, out_channels, kernel_size=4, stride=2, padding=1
        #     # ),
        #     torch.nn.ConvTranspose2d(
        #         in_channels, in_channels, kernel_size=4, stride=2, padding=1
        #     ),
        #     nonlinearity(),
        #     torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     # torch.nn.Tanh(),
        # )
        layers.append(torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decodes the latent into an observation.

        Args:
            latent: Encoded latent.

        Returns:
            Decoded observation.
        """
        # [B, latent_dim] => [B, 256, 2, 2].
        latent = latent.unsqueeze(-1).unsqueeze(-1)
        features = self.fc(latent)

        # [B, 256, 2, 2] => [B, 3, H, W].
        observation = self.decoder(features)

        return observation
