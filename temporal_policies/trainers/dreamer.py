import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from temporal_policies import (
    agents,
    datasets,
    dynamics as dynamics_module,
    envs,
    planners,
)
from temporal_policies.trainers.agents import AgentTrainer
from temporal_policies.trainers.daf import DafTrainer
from temporal_policies.utils import metrics, tensors
from temporal_policies.utils.typing import WrappedBatch


class DreamerTrainer(DafTrainer):
    """DAF-style unified trainer."""

    def __init__(
        self,
        env: envs.Env,
        eval_env: Optional[envs.Env],
        path: Union[str, pathlib.Path],
        dynamics: dynamics_module.LatentDynamics,
        planner: planners.Planner,
        dynamics_trainer_config: Union[str, pathlib.Path, Dict[str, Any]],
        agent_trainer_config: Union[str, pathlib.Path, Dict[str, Any]],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        env_kwargs: Dict[str, Any] = {},
        closed_loop_planning: bool = True,
        device: str = "auto",
        num_pretrain_steps: int = 1000,
        num_train_steps: int = 100000,
        num_eval_steps: int = 100,
        eval_freq: int = 1000,
        checkpoint_freq: int = 10000,
        log_freq: int = 100,
        profile_freq: Optional[int] = None,
    ):
        """Prepares the unified trainer for training.

        Args:
            path: Output path.
            dynamics: Dynamics model.
            planner: Planner for collect step.
            dynamics_trainer_config: Dynamics trainer config.
            agent_trainer_config: Agent trainer config.
            checkpoint: Optional path to trainer checkpoint.
            env_kwargs: Optional kwargs passed to EnvFactory.
            sample_primitive_actions: Whether to sample actions from the
                primitive distribution.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain. Overrides
                agent trainer configs.
            num_train_steps: Number of steps to train. Overrides agent and
                dynamics trainer configs.
            num_eval_steps: Number of steps per evaluation. Overrides agent and
                dynamics trianer configs.
            eval_freq: Evaluation frequency. Overrides agent and dynamics
                trainer configs.
            checkpoint_freq: Checkpoint frequency (separate from latest/best
                eval checkpoints).
            log_freq: Logging frequency. Overrides agent and dynamics trainer
                configs.
            profile_freq: Profiling frequency. Overrides agent and dynamics
                trainer configs.
        """
        super().__init__(
            env=env,
            eval_env=eval_env,
            path=path,
            dynamics=dynamics,
            planner=planner,
            dynamics_trainer_config=dynamics_trainer_config,
            agent_trainer_config=agent_trainer_config,
            checkpoint=checkpoint,
            env_kwargs=env_kwargs,
            closed_loop_planning=closed_loop_planning,
            device=device,
            num_pretrain_steps=num_pretrain_steps,
            num_train_steps=num_train_steps,
            num_eval_steps=num_eval_steps,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            log_freq=log_freq,
            profile_freq=profile_freq,
        )

        for agent_trainer in self.agent_trainers:
            if isinstance(agent_trainer.agent, agents.SAC):
                agent_trainer.agent.actor_update_freq == 0
            else:
                raise NotImplementedError

    def collect_agent_step(
        self,
        agent_trainer: AgentTrainer,
        env: envs.Env,
        action_skeleton: Sequence[envs.Primitive],
        action: np.ndarray,
        dataset: datasets.ReplayBuffer,
        t: int,
    ) -> Dict[str, Any]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics.
        """
        primitive = action_skeleton[0]
        env.set_primitive(primitive)
        observation = env.get_observation()
        dataset.add(observation=observation)
        agent_trainer._episode_length = 0
        agent_trainer._episode_reward = 0.0

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        discount = 1.0 - float(done)
        policy_args: Dict[str, Any] = {
            "remaining_plan": [
                {
                    "idx_policy": primitive.idx_policy,
                    "policy_args": primitive.get_policy_args(),
                }
                for primitive in action_skeleton[1:]
            ]
        }
        try:
            policy_args.update(info["policy_args"])
        except KeyError:
            pass

        dataset.add(
            action=action,
            reward=reward,
            next_observation=next_observation,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            policy_args=policy_args,
        )

        agent_trainer._episode_length += 1
        agent_trainer._episode_reward += reward
        if not done:
            return {}

        self.increment_epoch()

        step_metrics = {
            f"{metric}_{t}": value
            for metric, value in info.items()
            if metric in metrics.METRIC_AGGREGATION_FNS
        }
        step_metrics[f"reward_{t}"] = reward

        # Reset the environment
        self._episode_length = 0
        self._episode_reward = 0.0

        return step_metrics

    def plan_step(
        self,
        env: envs.Env,
        action_skeleton: Sequence[envs.Primitive],
        random: bool,
        mode: DafTrainer.Mode,
        return_full: bool = False,
    ) -> np.ndarray:
        if return_full:
            raise NotImplementedError("Dreamer is greedy")

        primitive = action_skeleton[0]
        if random:
            return primitive.sample()

        # Return greedy action.
        self.env.set_primitive(primitive)
        agent = self.agent_trainers[primitive.idx_policy].agent
        policy_args = primitive.get_policy_args()
        t_observation = tensors.from_numpy(env.get_observation(), self.device)
        t_observation = agent.encoder.encode(t_observation, policy_args)
        t_action = agent.actor.predict(
            t_observation, sample=mode == DafTrainer.Mode.COLLECT
        )
        action = t_action.detach().cpu().numpy()

        return action

    @tensors.vmap(dims=0)
    def rollout(
        self,
        env: envs.Env,
        plan_policy_args: Dict[str, List[Dict[str, Any]]],
        observation: torch.Tensor,
        dynamics_state: torch.Tensor,
    ) -> torch.Tensor:
        dynamics = self.dynamics_trainer.dynamics

        values = []
        for t, policy_args in enumerate(plan_policy_args["remaining_plan"]):
            primitive = env.get_primitive_info(**policy_args)

            policy = self.agent_trainers[primitive.idx_policy].agent
            policy_state = dynamics.decode(dynamics_state, primitive)

            # Get policy action.
            action = policy.actor.predict(policy_state)

            # Get policy value.
            value = policy.critic.predict(policy_state, action)
            values.append(value.unsqueeze(0))

            # Get next dynamics state.
            dynamics_state = dynamics.forward_eval(dynamics_state, action, primitive)

        values.append(
            torch.zeros(
                self._max_num_actions - len(plan_policy_args["remaining_plan"]),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
        )

        return torch.cat(values, dim=0)

    def compute_actors_loss(
        self,
        observation: torch.Tensor,
        reward: torch.Tensor,
        idx_policy: torch.Tensor,
        policy_args: np.ndarray,
    ) -> torch.Tensor:
        observation = observation.detach().requires_grad_()
        reward = reward.detach()
        dynamics = self.dynamics_trainer.dynamics
        dynamics.plan_mode()

        # Encode first dynamics state.
        dynamics_state = dynamics.encode(observation, idx_policy, policy_args)

        # Get imagined trajectory rewards.
        values = self.rollout(self.env, policy_args, observation, dynamics_state)

        # Maximize the imagined rewards.
        actors_loss = -values.min(dim=-1).values.sum()

        dynamics.train_mode()

        return actors_loss

    def actors_train_step(self, batch: WrappedBatch) -> Dict[str, Mapping[str, float]]:
        # Compute Dreamer actor loss.
        observation: torch.Tensor = batch["observation"]  # type: ignore
        reward: torch.Tensor = batch["reward"]  # type: ignore
        idx_policy: torch.Tensor = batch["idx_replay_buffer"]  # type: ignore
        policy_args = batch["policy_args"]
        actors_loss = self.compute_actors_loss(
            observation=observation,
            reward=reward,
            idx_policy=idx_policy,
            policy_args=batch["policy_args"],
        )
        metrics: Dict[str, Mapping[str, float]] = {
            self.dynamics_trainer.name: {"reward": -actors_loss.detach().item()}
        }

        # Update actor with Dreamer loss.
        for agent_trainer in self.agent_trainers:
            agent_trainer.optimizers["actor"].zero_grad(set_to_none=True)

        actors_loss.backward()

        for i, agent_trainer in enumerate(self.agent_trainers):
            policy = agent_trainer.agent
            assert isinstance(policy, agents.SAC)

            agent_trainer.optimizers["actor"].step()
            agent_trainer.schedulers["actor"].step()

            # Update alpha with SAC loss.
            ii = idx_policy == i
            _, alpha_loss, agent_metrics = policy.compute_actor_and_alpha_loss(
                policy.encoder.encode(observation[ii], policy_args[ii.cpu()])
            )
            metrics[agent_trainer.name] = agent_metrics

            agent_trainer.optimizers["log_alpha"].zero_grad(set_to_none=True)
            alpha_loss.backward()
            agent_trainer.optimizers["log_alpha"].step()
            agent_trainer.schedulers["log_alpha"].step()

        return metrics

    def train_step(  # type: ignore
        self, step: int, batch: WrappedBatch
    ) -> Dict[str, Mapping[str, float]]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.

        Returns:
            Dict of training metrics for each trainer for logging.
        """
        # Collect experience.
        collect_metrics = self.collect_step(random=False)

        # Train step.
        batch = tensors.to(batch, self.device)
        dynamics_metrics = self.dynamics_trainer.train_step(step, batch)
        actors_metrics = self.actors_train_step(batch)

        train_metrics = actors_metrics
        train_metrics[self.dynamics_trainer.name].update(dynamics_metrics)  # type: ignore
        for key in train_metrics:
            if key not in collect_metrics:
                collect_metrics[key] = {}

        return {
            key: {**collect_metrics[key], **train_metrics[key]} for key in train_metrics
        }
