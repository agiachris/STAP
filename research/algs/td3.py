import torch
import copy
import numpy as np
import itertools

from .base import Algorithm
from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_tensor, to_device, unsqueeze


class TD3(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       policy_noise=0.1,
                       target_noise=0.2,
                       noise_clip=0.5,
                       critic_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       init_steps=1000,
                       **kwargs):
        super().__init__(env, network_class, dataset_class, **kwargs)
        # Save extra parameters
        self.tau = tau
        self.policy_noise = policy_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.action_range = (self.env.action_space.low, self.env.action_space.high)
        self.action_range_tensor = to_device(to_tensor(self.action_range), self.device)
        self.init_steps = init_steps
        
        # Now setup the logging parameters
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0

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

    def _update_critic(self, batch):
        with torch.no_grad():
            noise = (torch.randn_like(batch['action']) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.target_network.actor(batch['next_obs'])
            noisy_next_action = (next_action + noise).clamp(*self.action_range_tensor)
            target_q1, target_q2 = self.target_network.critic(batch['next_obs'], noisy_next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = batch['reward'] + batch['discount']*target_q

        q1, q2 = self.network.critic(batch['obs'], batch['action'])
        q1_loss = torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        
        self.optim['critic'].zero_grad()
        q_loss.backward()
        self.optim['critic'].step()

        return dict(q1_loss=q1_loss.item(), q2_loss=q2_loss.item(), q_loss=q_loss.item(), target_q=target_q.mean().item())
    
    def _update_actor(self, batch):
        obs = batch['obs'].detach() # Detach the encoder so it isn't updated.
        action = self.network.actor(obs)
        q1, q2 = self.network.critic(obs, action)
        q = (q1 + q2) / 2
        actor_loss = -q.mean()

        self.optim['actor'].zero_grad()
        actor_loss.backward()
        self.optim['actor'].step()

        return dict(actor_loss=actor_loss.item())

    def _train_step(self, batch):
        all_metrics = {}
        if self.steps == 0: 
            self.dataset.add(self._current_obs) # Store the initial reset observation!
        if self.steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.network.eval()
            action = self.noisy_predict(self._current_obs)
            self.network.train()
        
        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward

        if 'discount' in info:
            discount = info['discount']
        elif hasattr(self.env, "_max_episode_stes") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences
        self.dataset.add(next_obs, action, reward, done, discount)
        
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
            metrics = self._update_actor(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def noisy_predict(self, obs):
        with torch.no_grad():
            action = self.predict(obs)
        action += self.policy_noise * np.random.randn(action.shape[0])
        return action
