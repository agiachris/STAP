from typing import Any, Optional, Union

import gym
import numpy as np
import torch

from temporal_policies import envs
from temporal_policies.networks.encoders.base import Encoder


class ResNet(Encoder):
    """ResNet encoder."""

    def __init__(
        self,
        env: envs.Env,
        out_features: int,
        variant: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        state_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(out_features,),
            dtype=np.float32,
        )
        super().__init__(env, state_space)

        if variant in ("resnet18", "resnet34"):
            dim_conv4_out = 256
        elif variant in ("resnet50", "resnet101", "resnet152"):
            dim_conv4_out = 1024
        else:
            raise NotImplementedError

        resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", variant, pretrained=pretrained
        )
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        # First four layers of ResNet (output of conv4).
        resnet_conv4 = list(resnet.children())[:-3]
        self.features = torch.nn.Sequential(*resnet_conv4)

        # Reduce to single pixel.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Output required feature dimensions.
        assert dim_conv4_out == out_features
        # if dim_conv4_out != out_features:
        #     self.fc = torch.nn.Linear(dim_conv4_out, out_features)
        # else:
        #     self.fc = torch.nn.Identity()

        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.img_mean = self.img_mean.unsqueeze(-1).unsqueeze(-1)
        self.img_stddev = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.img_stddev = self.img_stddev.unsqueeze(-1).unsqueeze(-1)

    def _apply(self, fn):
        super()._apply(fn)
        self.img_mean = fn(self.img_mean)
        self.img_stddev = fn(self.img_stddev)
        return self

    def forward(
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

        # Normalize pixels.
        if observation.dtype == torch.uint8:
            observation = (observation.float() / 255 - self.img_mean) / self.img_stddev

        # [B, 3, H, W] => [B, 512, H / 16, W / 16].
        x = self.features(observation)

        # [B, 512, H / 16, W / 16] => [B, conv4_out].
        x = self.avgpool(x).squeeze(-1).squeeze(-1)

        # [B, conv4_out, 1, 1] => [B, out_features].
        # x = self.fc(x)

        # Restore original input dimensions.
        if squeeze:
            x = x.squeeze(0)

        return x
