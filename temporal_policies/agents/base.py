from typing import Union

import gym
import torch

from temporal_policies import encoders, networks
from temporal_policies.utils import tensors


class Agent:
    """Base agent class."""

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        observation_space: gym.spaces.Box,
        actor: networks.actors.Actor,
        critic: networks.critics.Critic,
        encoder: encoders.Encoder,
        device: str = "auto",
    ):
        """Assigns the required properties of the Agent.

        Args:
            state_space: Policy state space (encoder output, actor/critic input).
            action_space: Action space (actor output).
            observation_space: Observation space (encoder input).
            actor: Actor network.
            critic: Critic network.
            encoder: Encoder network.
            device: Torch device.
        """
        assert isinstance(action_space, gym.spaces.Box)
        self._state_space = state_space
        self._action_space = action_space
        self._observation_space = observation_space
        self._actor = actor
        self._critic = critic
        self._encoder = encoder
        self.to(device)

    @property
    def state_space(self) -> gym.spaces.Box:
        """Policy state space (encoder output, actor/critic input)."""
        return self._state_space

    @property
    def action_space(self) -> gym.spaces.Box:
        """Action space (actor output)."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Observation space (encoder input)."""
        return self._observation_space

    @property
    def actor(self) -> networks.actors.Actor:
        """Actor network that takes as input a state and outputs an action."""
        return self._actor

    @property
    def critic(self) -> networks.critics.Critic:
        """Critic network that takes as input a state/action and outputs a
        success probability."""
        return self._critic

    @property
    def encoder(self) -> encoders.Encoder:
        """Encoder network that encodes observations into states."""
        return self._encoder

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Agent":
        """Transfers networks to a device."""
        self._device = tensors.device(device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.encoder.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches the networks to train mode."""
        self.actor.train()
        self.critic.train()
        self.encoder.train_mode()

    def eval_mode(self) -> None:
        """Switches the networks to eval mode."""
        self.actor.eval()
        self.critic.eval()
        self.encoder.eval_mode()
