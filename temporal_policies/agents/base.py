import gym  # type: ignore
import torch  # type: ignore


class Agent:
    """Base agent class."""

    def __init__(
        self,
        state_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        encoder: torch.nn.Module,
    ):
        """Assigns the required properties of the Agent.

        Args:
            state_space: State space (encoder output, actor/critic input).
            action_space: Action space (actor output).
            observation_space: Observation space (encoder input).
            actor: Actor network.
            critic: Critic network.
            encoder: Encoder network.
        """
        self._state_space = state_space
        self._action_space = action_space
        self._observation_space = observation_space
        self._actor = actor
        self._critic = critic
        self._encoder = encoder

    @property
    def state_space(self) -> gym.spaces.Space:
        """State space (encoder output, actor/critic input)."""
        return self._state_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """Action space (actor output)."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Observation space (encoder input)."""
        return self._observation_space

    @property
    def actor(self) -> torch.nn.Module:
        """Actor network that takes as input a state and outputs an action."""
        return self._actor

    @property
    def critic(self) -> torch.nn.Module:
        """Critic network that takes as input a state/action and outputs a
        success probability."""
        return self._critic

    @property
    def encoder(self) -> torch.nn.Module:
        """Encoder network that encodes observations into states."""
        return self._encoder

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self.actor.device

    def to(self, device: torch.device) -> "Agent":
        """Transfer networks to a device."""
        self.actor.to(device)
        self.critic.to(device)
        self.encoder.to(device)
        return self
