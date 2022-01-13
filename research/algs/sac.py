import torch
import copy
import numpy as np
import itertools

from .base import Algorithm
from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_tensor, to_device, unsqueeze


class SAC(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       init_temperature=0.1,
                       critic_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       init_steps=1000,
                       **kwargs):

        # Save values needed for optim setup.
        self.init_temperature = init_temperature

        super().__init__(env, network_class, dataset_class, **kwargs)
        # Save extra parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.init_steps = init_steps

        # Now setup the logging parameters
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def setup_network(self, network_class, network_kwargs):
        self.network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Default optimizer initialization
        self.optim['actor'] = optim_class(self.network.actor.parameters(), **optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())        
        self.optim['critic'] = optim_class(critic_params, **optim_kwargs)

        # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.env.action_space.low.shape)

        self.optim['log_alpha'] = optim_class([self.log_alpha], **optim_kwargs)

    def _update_critic(self, batch):
        with torch.no_grad():
            dist = self.network.actor(batch['next_obs'])
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1)
            target_q1, target_q2 = self.target_network.critic(batch['next_obs'], next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob
            target_q = batch['reward'] + batch['discount']*target_v

        q1, q2 = self.network.critic(batch['obs'], batch['action'])
        q1_loss = torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.optim['critic'].zero_grad()
        q_loss.backward()
        self.optim['critic'].step()

        return dict(q1_loss=q1_loss.item(), q2_loss=q2_loss.item(), q_loss=q_loss.item(), 
                    target_q=target_q.mean().item())
    
    def _update_actor_and_alpha(self, batch):
        obs = batch['obs'].detach() # Detach the encoder so it isn't updated.
        dist = self.network.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        q1, q2 = self.network.critic(obs, action)
        q = torch.min(q1, q2)
        assert log_prob.shape == q.shape
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.optim['actor'].zero_grad()
        actor_loss.backward()
        self.optim['actor'].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim['log_alpha'].zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.optim['log_alpha'].step()

        return dict(actor_loss=actor_loss.item(), entropy=entropy.item(), 
                    alpha_loss=alpha_loss.item(), alpha=self.alpha.detach().item())

    def _train_step(self, batch):
        all_metrics = {}
        if self.steps == 0: 
            self.dataset.add(self._current_obs) # Store the initial reset observation!
        if self.steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.network.eval()
            with torch.no_grad():
                obs = torch.FloatTensor(self._current_obs.copy()).to(self.device)
                obs = obs.unsqueeze(0)
                dist = self.network.actor(obs)
                action = dist.sample()
                action = action.clamp(-1, 1)
                action = action[0].cpu().numpy()
            self.network.train()

            # self.network.eval()
            # action = self.predict(self._current_obs, sample=True)
            # self.network.train()
        
        next_obs, reward, done, _ = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward
        # Store the consequences
        self.dataset.add(next_obs, action, reward, done)
        
        if done:
            self._num_ep += 1
            # update metrics
            all_metrics['reward'] = self._episode_reward
            all_metrics['length'] = self._episode_length
            all_metrics['num_ep'] = self._num_ep
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs) # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs
        
        if self.steps < self.init_steps or not 'obs' in batch:
            return all_metrics

        updating_critic = self.steps % self.critic_freq == 0
        updating_actor = self.steps % self.actor_freq == 0

        if updating_actor or updating_critic:
            batch['obs'] = self.network.encoder(batch['obs'])
            with torch.no_grad():
                batch['next_obs'] = self.target_network.encoder(batch['next_obs'])
        
        if updating_critic:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if updating_actor:
            metrics = self._update_actor_and_alpha(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.encoder.parameters(), self.target_network.encoder.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.critic.parameters(), self.target_network.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")
