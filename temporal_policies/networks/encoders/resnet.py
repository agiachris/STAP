import torch  # type: ignore

from temporal_policies.networks.encoders.base import Encoder


class ResNet(Encoder):
    """ResNet encoder."""

    def __init__(
        self,
        env,
        out_features: int,
        variant: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
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
        self.fc = torch.nn.Linear(512, out_features)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.features(observation)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
