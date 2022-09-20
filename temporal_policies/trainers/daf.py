import enum
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import tqdm

from temporal_policies import datasets, dynamics as dynamics_module, envs, planners
from temporal_policies.trainers.agents import AgentTrainer
from temporal_policies.trainers.base import Trainer
from temporal_policies.trainers.unified import UnifiedTrainer
from temporal_policies.utils import metrics, spaces, tensors
from temporal_policies.utils.typing import Batch, Scalar, WrappedBatch


class DafTrainer(UnifiedTrainer):
    """DAF-style unified trainer."""

    class Mode(enum.Enum):
        COLLECT = 0
        EVALUATE = 1

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
            closed_loop_planning: Whether to use closed-loop planning.
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
            path=path,
            dynamics=dynamics,
            dynamics_trainer_config=dynamics_trainer_config,
            agent_trainer_config=agent_trainer_config,
            checkpoint=checkpoint,
            env_kwargs=env_kwargs,
            device=device,
            num_pretrain_steps=num_pretrain_steps,
            num_train_steps=num_train_steps,
            num_eval_steps=num_eval_steps,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            log_freq=log_freq,
            profile_freq=profile_freq,
        )

        self._env = env
        self._eval_env = env if eval_env is None else eval_env
        self._planner = planner

        self.closed_loop_planning = closed_loop_planning
        assert isinstance(self.env, envs.pybullet.TableEnv)
        self._max_num_actions = max(
            len(task.action_skeleton) for task in self.env.tasks.tasks
        )

    @property
    def env(self) -> envs.Env:
        return self._env

    @property
    def eval_env(self) -> envs.Env:
        return self._eval_env

    @property
    def planner(self) -> planners.Planner:
        """Planner used during collect step."""
        return self._planner

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
        try:
            policy_args = info["policy_args"]
        except KeyError:
            policy_args = None

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
        # step_metrics[f"length_{t}"] = agent_trainer._episode_length
        # step_metrics[f"episode_{t}"] = agent_trainer.epoch

        # Reset the environment
        self._episode_length = 0
        self._episode_reward = 0.0

        return step_metrics

    def plan_step(
        self,
        env: envs.Env,
        action_skeleton: Sequence[envs.Primitive],
        random: bool,
        mode: Mode,
        return_full: bool = False,
    ) -> np.ndarray:
        if random:
            if return_full:
                raise NotImplementedError("Primitive sampling is state-dependent")
            return action_skeleton[0].sample()

        with self.dynamics_trainer.profiler.profile("plan"):
            # Plan.
            env.set_primitive(action_skeleton[0])
            observation = env.get_observation()
            actions = self.planner.plan(
                observation=observation, action_skeleton=action_skeleton
            ).actions

        return actions if return_full else actions[0]

    def collect_step(
        self, random: bool = False, mode: Mode = Mode.COLLECT
    ) -> Dict[str, Mapping[str, float]]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics for each trainer.
        """
        env = self.eval_env if mode == DafTrainer.Mode.EVALUATE else self.env
        self.eval_mode()
        self.dynamics_trainer.dynamics.plan_mode()
        env.reset()

        if not self.closed_loop_planning and not random:
            actions = self.plan_step(
                env, env.action_skeleton, random=random, mode=mode, return_full=True
            )

        failure = False
        collect_metrics: Dict[str, Dict[str, float]] = {
            trainer.name: {f"reward_{t}": 0.0 for t in range(self._max_num_actions)}
            for trainer in self.agent_trainers
        }
        for t, primitive in enumerate(env.action_skeleton):
            agent_trainer = self.agent_trainers[primitive.idx_policy]
            with agent_trainer.profiler.profile(mode.name.lower()):
                if failure:
                    collect_metrics[agent_trainer.name].update(
                        {
                            f"reward_{t}": 0.0,
                            # f"length_{t}": 0,
                            # f"episode_{t}": agent_trainer.epoch,
                        }
                    )
                    continue

                # Plan.
                if self.closed_loop_planning or random:
                    action = self.plan_step(
                        env, env.action_skeleton[t:], random=random, mode=mode
                    )
                else:
                    action = actions[t]

                # Execute first step.
                dataset = (
                    agent_trainer.eval_dataset
                    if mode == "evaluate"
                    else agent_trainer.dataset
                )
                collect_metrics[agent_trainer.name].update(
                    self.collect_agent_step(
                        agent_trainer, env, env.action_skeleton[t:], action, dataset, t
                    )
                )
                if collect_metrics[agent_trainer.name][f"reward_{t}"] == 0.0:
                    failure = True

        for trainer in self.agent_trainers:
            if trainer.name not in collect_metrics:
                collect_metrics[trainer.name] = {agent_trainer.eval_metric: 0.0}
                continue

            reward = sum(
                reward
                for key, reward in collect_metrics[trainer.name].items()
                if agent_trainer.eval_metric in key
            )
            collect_metrics[trainer.name][agent_trainer.eval_metric] = reward
        collect_metrics[self.dynamics_trainer.name] = {}

        self.train_mode()

        return collect_metrics  # type: ignore

    def evaluate_step(self) -> Dict[str, Mapping[str, float]]:
        return self.collect_step(random=False, mode=DafTrainer.Mode.EVALUATE)

    def evaluate(self) -> Dict[str, List[Mapping[str, Union[Scalar, np.ndarray]]]]:  # type: ignore
        """Evaluates the policies and dynamics model.

        Returns:
            Eval metrics for each trainer.
        """
        self.eval_mode()

        eval_metrics_list: Dict[str, List[Mapping[str, Union[Scalar, np.ndarray]]]] = {
            trainer.name: [] for trainer in self.trainers
        }
        pbar = tqdm.tqdm(
            range(self.num_eval_steps),
            desc=f"Eval {self.name}",
            dynamic_ncols=True,
        )
        for _ in pbar:
            eval_metrics = self.evaluate_step()
            for key, eval_metric in eval_metrics.items():
                eval_metrics_list[key].append(eval_metric)
            pbar.set_postfix(
                {
                    f"{trainer.name}/{trainer.eval_metric}": eval_metrics[trainer.name][
                        trainer.eval_metric
                    ]
                    for trainer in self.agent_trainers
                    if trainer.name in eval_metrics
                }
            )

        eval_metrics_list[self.dynamics_trainer.name] = self.dynamics_trainer.evaluate()

        for agent_trainer in self.agent_trainers:
            agent_trainer.eval_dataset.save()

        self.train_mode()

        return eval_metrics_list

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
        train_metrics = {}
        for idx_policy, agent_trainer in enumerate(self.agent_trainers):
            idx_batch = batch["idx_replay_buffer"] == idx_policy
            agent_batch: Batch = tensors.map_structure(  # type: ignore
                lambda x: x[idx_batch],
                {key: val for key, val in batch.items() if key != "idx_replay_buffer"},
            )

            # Torch dataloader makes it a tensor?
            assert isinstance(agent_batch["action"], torch.Tensor)
            agent_batch["action"] = spaces.subspace(
                agent_batch["action"].numpy(), agent_trainer.agent.action_space
            )
            # Do not collect data during agent train step.
            agent_train_metrics = Trainer.train_step(agent_trainer, step, agent_batch)
            train_metrics[agent_trainer.name] = agent_train_metrics

        dynamics_train_metrics = self.dynamics_trainer.train_step(step, batch)
        train_metrics[self.dynamics_trainer.name] = dynamics_train_metrics

        for key in train_metrics:
            if key not in collect_metrics:
                collect_metrics[key] = {}
        return {
            key: {**collect_metrics[key], **train_metrics[key]} for key in train_metrics
        }
