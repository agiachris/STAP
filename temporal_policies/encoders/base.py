from typing import Any, Dict, Generic, Type, Union

import gym
import torch

from temporal_policies import envs, networks
from temporal_policies.utils import configs, tensors
from temporal_policies.utils.typing import ObsType


class Encoder(Generic[ObsType]):
    """Base encooder class."""

    def __init__(
        self,
        env: envs.Env,
        network_class: Union[str, Type[networks.encoders.Encoder]],
        network_kwargs: Dict[str, Any],
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            env: Encoder env.
            network_class: Dynamics model network class.
            network_kwargs: Kwargs for network class.
            device: Torch device.
        """
        network_class = configs.get_class(network_class, networks)
        self._network = network_class(env, **network_kwargs)

        self._observation_space = env.observation_space

        self.to(device)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def state_space(self) -> gym.spaces.Space:
        return self.network.state_space

    @property
    def network(self) -> networks.encoders.Encoder:
        return self._network

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Encoder[ObsType]":
        """Transfers networks to device."""
        self._device = torch.device(tensors.device(device))
        self.network.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches the networks to train mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Switches the networks to eval mode."""
        self.network.eval()

    def encode(self, observation: ObsType) -> torch.Tensor:
        return self.network.predict(observation)
