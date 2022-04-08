import gym  # type: ignore
import torch  # type: ignore

import temporal_policies


class NormalizeObservation(torch.nn.Module):
    """Normalizes observation to the range (-0.5, 0.5)."""

    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__()
        self.observation_mid = torch.from_numpy(
            (observation_space.low + observation_space.high) / 2
        )
        self.observation_range = torch.from_numpy(
            observation_space.high - observation_space.low
        )

    def _apply(self, fn):
        """Ensures members get transferred with NormalizeObservation.to(device)."""
        super()._apply(fn)
        self.observation_mid = fn(self.observation_mid)
        self.observation_range = fn(self.observation_range)
        return self

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalizes observation to the range (-0.5, 0.5)."""
        return (observation - self.observation_mid) / self.observation_range


class ActorCriticPolicy(torch.nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        actor_class,
        critic_class,
        encoder_class=None,
        actor_kwargs={},
        critic_kwargs={},
        encoder_kwargs={},
        **kwargs
    ) -> None:
        super().__init__()
        encoder_class = (
            vars(temporal_policies.networks)[encoder_class]
            if isinstance(encoder_class, str)
            else encoder_class
        )
        actor_class = (
            vars(temporal_policies.networks)[actor_class]
            if isinstance(actor_class, str)
            else actor_class
        )
        critic_class = (
            vars(temporal_policies.networks)[critic_class]
            if isinstance(critic_class, str)
            else critic_class
        )

        encoder_kwargs.update(kwargs)
        actor_kwargs.update(kwargs)
        critic_kwargs.update(kwargs)

        if encoder_class is not None:
            self._encoder = encoder_class(
                observation_space, action_space, **encoder_kwargs
            )
            # Modify the observation space
            if hasattr(self._encoder, "output_space"):
                observation_space = self._encoder.output_space
        else:
            # self._encoder = torch.nn.Identity()
            self._encoder = NormalizeObservation(observation_space)
        self._actor = actor_class(observation_space, action_space, **actor_kwargs)
        self._critic = critic_class(observation_space, action_space, **critic_kwargs)

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic

    @property
    def encoder(self):
        return self._encoder

    def predict(self, obs, encoded=False, **kwargs):
        if not encoded:
            obs = self._encoder(obs)
        if hasattr(self._actor, "predict"):
            return self._actor.predict(obs, **kwargs)
        else:
            return self._actor(obs)
