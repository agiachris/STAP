from abc import ABC, abstractmethod

class ActorCriticPolicy(ABC):

    @property
    @abstractmethod
    def actor(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def critic(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs):
        raise NotImplementedError
