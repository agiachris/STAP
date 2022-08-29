import torch
import numpy as np
import itertools

from temporal_policies.agents import rl
from temporal_policies.networks.base import ActorCriticPolicy


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.distributions.utils._standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class DRQV2(rl.RLAgent):
    def __init__(
        self,
        env,
        network_class,
        dataset_class,
        tau=0.005,
        critic_freq=1,
        actor_freq=1,
        target_freq=1,
        init_steps=1000,
        std_schedule=(1.0, 0.1, 500000),
        noise_clip=0.3,
        **kwargs
    ):
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)
        # Save extra parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.std_schedule = std_schedule
        self.init_steps = init_steps
        self.noise_clip = noise_clip

        # Now setup the logging parameters
        self._current_obs, _ = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0

    def setup_network(self, network_class, network_kwargs):
        self.network = network_class(
            self.env.observation_space, self.env.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.env.observation_space, self.env.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Default optimizer initialization
        self.optim["actor"] = optim_class(
            self.network.actor.parameters(), **optim_kwargs
        )
        # Update the encoder with the critic.
        critic_params = itertools.chain(
            self.network.critic.parameters(), self.network.encoder.parameters()
        )
        self.optim["critic"] = optim_class(critic_params, **optim_kwargs)

    def _update_critic(self, batch):
        with torch.no_grad():
            mu = self.network.actor(batch["next_observation"])
            std = self._get_std() * torch.ones_like(mu)
            next_action = TruncatedNormal(mu, std).sample(clip=self.noise_clip)
            target_q1, target_q2 = self.target_network.critic(
                batch["next_observation"], next_action
            )
            target_v = torch.min(target_q1, target_q2)
            target_q = batch["reward"] + batch["discount"] * target_v

        q1, q2 = self.network.critic(batch["observation"], batch["action"])
        q1_loss = torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(
            q1_loss=q1_loss.item(),
            q2_loss=q2_loss.item(),
            q_loss=q_loss.item(),
            target_q=target_q.mean().item(),
        )

    def _update_actor(self, batch):
        obs = batch["observation"].detach()  # Detach the encoder so it isn't updated.
        mu = self.network.actor(obs)
        std = self._get_std() * torch.ones_like(mu)
        dist = TruncatedNormal(mu, std)
        action = dist.sample(clip=self.noise_clip)
        log_prob = dist.log_prob(action).sum(dim=-1)

        q1, q2 = self.network.critic(obs, action)
        q = torch.min(q1, q2)
        actor_loss = -q.mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        return dict(actor_loss=actor_loss.item(), log_prob=log_prob.mean().item())

    def _train_step(self, batch):
        all_metrics = {}
        if self.steps == 0:
            self.dataset.add(self._current_obs)  # Store the initial reset observation!
        if self.steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                mu = self.predict(self._current_obs)
                mu = torch.as_tensor(mu, device=self.device)
                std = self._get_std() * torch.ones_like(mu)
                action = TruncatedNormal(mu, std).sample(clip=None).cpu().numpy()
            self.train_mode()

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif (
            hasattr(self.env, "_max_episode_stes")
            and self._episode_length == self.env._max_episode_steps
        ):
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences
        self.dataset.add(next_obs, action, reward, terminated, truncated, discount)

        if done:
            self._num_ep += 1
            # update metrics
            all_metrics["reward"] = self._episode_reward
            all_metrics["length"] = self._episode_length
            all_metrics["num_ep"] = self._num_ep
            # Reset the environment
            self._current_obs, _ = self.env.reset()
            self.dataset.add(self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        if self.steps < self.init_steps or "observation" not in batch:
            return all_metrics

        updating_critic = self.steps % self.critic_freq == 0
        updating_actor = self.steps % self.actor_freq == 0

        if updating_actor or updating_critic:
            batch["observation"] = self.network.encoder(batch["observation"])
            with torch.no_grad():
                batch["next_observation"] = self.network.encoder(
                    batch["next_observation"]
                )

        if updating_critic:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if updating_actor:
            metrics = self._update_actor(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            with torch.no_grad():
                # Only update the critic, don't update the encoder.
                for param, target_param in zip(
                    self.network.critic.parameters(),
                    self.target_network.critic.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        return all_metrics

    def _get_std(self):
        init, final, duration = self.std_schedule
        mix = np.clip(self.steps / duration, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")
