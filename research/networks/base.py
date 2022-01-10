from torch import nn
import research

class ActorCriticPolicy(nn.Module):

    def __init__(self, observation_space, action_space, 
                       actor_class, critic_class, encoder_class=None, 
                       actor_kwargs={}, critic_kwargs={}, encoder_kwargs={}, **kwargs) -> None:
        super().__init__()
        encoder_class = vars(research.networks)[encoder_class] if isinstance(encoder_class, str) else encoder_class
        actor_class = vars(research.networks)[actor_class] if isinstance(actor_class, str) else actor_class
        critic_class = vars(research.networks)[critic_class] if isinstance(critic_class, str) else critic_class

        encoder_kwargs.update(kwargs)
        actor_kwargs.update(kwargs)
        critic_kwargs.update(kwargs)

        if encoder_class is not None:
            self._encoder = encoder_class(observation_space, action_space, **encoder_kwargs)
            # Modify the observation space
            if hasattr(self._encoder, "output_space"):
                observation_space = self._encoder.output_space
        else:
            self._encoder = nn.Identity()
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
        
    def predict(self, obs, **kwargs):
        obs = self._encoder(obs)
        if hasattr(self._actor, "predict"):
            return self._actor.predict(obs, **kwargs)
        else:
            return self._actor(obs)
