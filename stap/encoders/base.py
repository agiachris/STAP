from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch

from stap import envs, networks
from stap.utils import configs, tensors


class Encoder:
    """Base encooder class."""

    def __init__(
        self,
        env: envs.Env,
        network_class: Union[str, Type[networks.encoders.Encoder]],
        network_kwargs: Dict[str, Any] = {},
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
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def state_space(self) -> gym.spaces.Box:
        return self.network.state_space

    @property
    def network(self) -> networks.encoders.Encoder:
        return self._network

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Encoder":
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

    def encode(
        self,
        observation: torch.Tensor,
        policy_args: Union[np.ndarray, Optional[Any]],
        **kwargs,
    ) -> torch.Tensor:
        return self.network.predict(observation, policy_args, **kwargs)
