import torch
import copy
import numpy as np

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
        self.optim['critic'] = optim_class(self.network.critic.parameters(), **optim_kwargs)

    def _compute_q_loss(self, batch):
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
        return q_loss, dict(q1_loss=q1_loss.item(), q2_loss=q2_loss.item(), q_loss=q_loss.item(), target_q=target_q.mean().item())
    
    def _compute_actor_loss(self, batch):
        action = self.network.actor(batch['obs'])
        q1, q2 = self.network.critic(batch['obs'], action)
        q = (q1 + q2) / 2
        actor_loss = -q.mean()
        return actor_loss, dict(actor_loss=actor_loss.item())

    def _train_step(self, batch):
        all_metrics = {}
        if self.steps == 0: 
            self.dataset.add(self._current_obs) # Store the initial reset observation!
        if self.steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            action = self.noisy_predict(self._current_obs)
        
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
        
        if self.steps < self.init_steps:
            return all_metrics
        
        if self.steps % self.critic_freq == 0:
            self.optim['critic'].zero_grad()
            loss, metrics = self._compute_q_loss(batch)
            loss.backward()
            self.optim['critic'].step()
            all_metrics.update(metrics)

        if self.steps % self.actor_freq == 0:
            self.optim['actor'].zero_grad()
            loss, metrics = self._compute_actor_loss(batch)
            loss.backward()
            self.optim['actor'].step()
            all_metrics.update(metrics)

            # update the target network
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch):
        '''
        perform a validation step
        '''
        all_metrics = {}
        with torch.no_grad():
            _, metrics = self._compute_q_loss(batch)
            all_metrics.update(metrics)
            _, metrics = self._compute_actor_loss(batch)
            all_metrics.update(metrics)
        return all_metrics

    def noisy_predict(self, obs):
        with torch.no_grad():
            action = self.predict(obs)
        action += self.policy_noise * np.random.randn(action.shape[0])
        return action

