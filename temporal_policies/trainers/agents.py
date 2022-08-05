import pathlib
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import tqdm

from temporal_policies import agents, datasets, envs, processors
from temporal_policies.networks.encoders import BASIC_ENCODERS
from temporal_policies.schedulers import DummyScheduler
from temporal_policies.trainers.base import Trainer
from temporal_policies.utils import configs, metrics, tensors
from temporal_policies.utils.typing import Batch, Scalar


class AgentTrainer(Trainer[agents.RLAgent, Batch, Batch]):
    """Agent trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        agent: agents.RLAgent,
        eval_env: Optional[envs.Env] = None,
        dataset_class: Union[str, Type[datasets.ReplayBuffer]] = datasets.ReplayBuffer,
        dataset_kwargs: Dict[str, Any] = {},
        eval_dataset_kwargs: Optional[Dict[str, Any]] = None,
        processor_class: Union[
            str, Type[processors.Processor]
        ] = processors.IdentityProcessor,
        processor_kwargs: Dict[str, Any] = {},
        optimizer_class: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
        scheduler_class: Union[
            str, Type[torch.optim.lr_scheduler._LRScheduler]
        ] = DummyScheduler,
        scheduler_kwargs: Dict[str, Any] = {},
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        num_pretrain_steps: int = 1000,
        num_train_steps: int = 100000,
        num_eval_episodes: int = 100,
        eval_freq: int = 1000,
        checkpoint_freq: int = 10000,
        log_freq: int = 100,
        profile_freq: Optional[int] = None,
        eval_metric: str = "reward",
        num_data_workers: int = 0,
        dataset_size: Optional[int] = None,
        eval_dataset_size: Optional[int] = None,
    ):
        """Prepares the agent trainer for training.

        Args:
            path: Training output path.
            agent: Agent to be trained.
            eval_env: Optional env for evaluation. If None, uses the agent's
                training environment for evaluation.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            eval_dataset_kwargs: Kwargs for eval dataset.
            processor_class: Batch data processor class.
            processor_kwargs: Kwargs for processor.
            optimizer_class: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler_class: Optional optimizer scheduler class.
            scheduler_kwargs: Kwargs for scheduler class.
            checkpoint: Optional path to trainer checkpoint.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain.
            num_train_steps: Number of steps to train.
            num_eval_episodes: Number of episodes per evaluation.
            eval_freq: Evaluation frequency.
            checkpoint_freq: Checkpoint frequency (separate from latest/best
                eval checkpoints).
            log_freq: Logging frequency.
            profile_freq: Profiling frequency.
            eval_metric: Metric to use for evaluation.
            num_data_workers: Number of workers to use for dataloader.
        """
        path = pathlib.Path(path) / agent.env.name

        dataset_class = configs.get_class(dataset_class, datasets)
        dataset_kwargs = dict(dataset_kwargs)
        dataset_kwargs["path"] = path / "train_data"
        dataset = dataset_class(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            **dataset_kwargs,
        )

        if eval_dataset_kwargs is None:
            eval_dataset_kwargs = dataset_kwargs
        eval_dataset_kwargs = dict(eval_dataset_kwargs)
        eval_dataset_kwargs["save_frequency"] = None
        eval_dataset_kwargs["path"] = path / "eval_data"
        eval_dataset = dataset_class(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            **eval_dataset_kwargs,
        )

        processor_class = configs.get_class(processor_class, processors)
        processor = processor_class(agent.observation_space, **processor_kwargs)

        optimizer_class = configs.get_class(optimizer_class, torch.optim)
        optimizers = agent.create_optimizers(optimizer_class, optimizer_kwargs)

        scheduler_class = configs.get_class(scheduler_class, torch.optim.lr_scheduler)
        schedulers = {
            key: scheduler_class(optimizer, **scheduler_kwargs)
            for key, optimizer in optimizers.items()
        }

        super().__init__(
            path=path,
            model=agent,
            dataset=dataset,
            eval_dataset=eval_dataset,
            processor=processor,
            optimizers=optimizers,
            schedulers=schedulers,
            checkpoint=None,
            device=device,
            num_pretrain_steps=num_pretrain_steps,
            num_train_steps=num_train_steps,
            num_eval_steps=num_eval_episodes,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            log_freq=log_freq,
            profile_freq=profile_freq,
            eval_metric=eval_metric,
            num_data_workers=num_data_workers,
        )

        if checkpoint is not None:
            self.load(
                checkpoint,
                strict=True,
                dataset_size=dataset_size,
                eval_dataset_size=eval_dataset_size,
            )

        self._eval_env = self.agent.env if eval_env is None else eval_env
        self._reset_collect = True

    @property
    def agent(self) -> agents.RLAgent:
        """Agent being trained."""
        return self.model

    @property
    def env(self) -> envs.Env:
        """Agent env."""
        return self.agent.env

    @property
    def eval_env(self) -> envs.Env:
        """Agent env."""
        return self._eval_env

    def collect_step(self, random: bool = False) -> Dict[str, Any]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics.
        """
        with self.profiler.profile("collect"):
            if self._reset_collect:
                self._reset_collect = False
                observation = self.env.reset()
                assert isinstance(observation, np.ndarray)
                self.dataset.add(observation=observation)
                self._episode_length = 0
                self._episode_reward = 0.0

            if random:
                action = self.agent.action_space.sample()
            else:
                self.eval_mode()
                with torch.no_grad():
                    t_observation = tensors.from_numpy(
                        self.env.get_observation(), self.device
                    )
                    if not isinstance(self.agent.encoder.network, BASIC_ENCODERS):
                        t_observation = tensors.rgb_to_cnn(t_observation)
                    t_action = self.agent.actor.predict(
                        self.agent.encoder.encode(t_observation)
                    )
                    action = t_action.cpu().numpy()
                self.train_mode()

            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            done = terminated or truncated
            discount = 1.0 - float(done)

            self.dataset.add(
                action=action,
                reward=reward,
                next_observation=next_observation,
                discount=discount,
                terminated=terminated,
                truncated=truncated,
            )

            self._episode_length += 1
            self._episode_reward += reward
            if not done:
                return {}

            self.increment_epoch()

            metrics = {
                "reward": self._episode_reward,
                "length": self._episode_length,
                "episode": self.epoch,
            }

            # Reset the environment
            observation = self.env.reset()
            assert isinstance(observation, np.ndarray)
            self.dataset.add(observation=observation)
            self._episode_length = 0
            self._episode_reward = 0.0

            return metrics

    def process_batch(self, batch: Batch) -> Batch:
        """Processes replay buffer batch for training.

        Args:
            batch: Replay buffer batch.

        Returns:
            Processed batch.
        """
        if (
            isinstance(batch["observation"], torch.Tensor)
            and batch["observation"].shape[-1] == 3
            and batch["observation"].dtype == torch.uint8
            and not isinstance(self.agent.encoder.network, BASIC_ENCODERS)
        ):
            batch["observation"] = tensors.rgb_to_cnn(batch["observation"])  # type: ignore
            batch["next_observation"] = tensors.rgb_to_cnn(batch["next_observation"])  # type: ignore
        return super().process_batch(batch)

    def pretrain(self) -> None:
        """Runs the pretrain phase."""
        self.dataset.initialize()
        pbar = tqdm.tqdm(
            range(self.step, self.num_pretrain_steps),
            desc=f"Pretrain {self.name}",
            dynamic_ncols=True,
        )
        metrics_list = []
        for _ in pbar:
            collect_metrics = self.collect_step(random=True)
            pbar.set_postfix({self.eval_metric: collect_metrics[self.eval_metric]})

            metrics_list.append(collect_metrics)
            metrics_list = self.log_step(metrics_list, stage="pretrain")

            self.increment_step()

    def train_step(self, step: int, batch: Batch) -> Dict[str, float]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.

        Returns:
            Dict of training metrics for logging.
        """
        collect_metrics = self.collect_step(random=False)
        train_metrics = super().train_step(step, batch)

        return {**collect_metrics, **train_metrics}

    def train(self) -> None:
        """Trains the model."""
        super().train()
        self.dataset.save()

    def evaluate(self) -> List[Dict[str, Union[Scalar, np.ndarray]]]:
        """Evaluates the model.

        Returns:
            Eval metrics.
        """
        self.eval_mode()

        with self.profiler.profile("evaluate"):
            metrics_list: List[Dict[str, Union[Scalar, np.ndarray]]] = []
            pbar = tqdm.tqdm(
                range(self.num_eval_steps),
                desc=f"Eval {self.name}",
                dynamic_ncols=True,
            )
            for _ in pbar:
                episode_metrics = self.evaluate_step()
                metrics_list.append(episode_metrics)
                pbar.set_postfix({self.eval_metric: episode_metrics[self.eval_metric]})

            self.eval_dataset.save()

        self.train_mode()
        self._reset_collect = True

        return metrics_list

    def evaluate_step(self) -> Dict[str, Union[Scalar, np.ndarray]]:
        """Performs a single evaluation step.

        Returns:
            Dict of eval metrics for one episode.
        """
        observation = self.eval_env.reset()
        assert isinstance(observation, np.ndarray)
        self.eval_dataset.add(observation=observation)

        step_metrics_list = []
        done = False
        while not done:
            with torch.no_grad():
                t_observation = tensors.from_numpy(observation, self.device)
                if not isinstance(self.agent.encoder.network, BASIC_ENCODERS):
                    t_observation = tensors.rgb_to_cnn(t_observation)
                t_action = self.agent.actor.predict(
                    self.agent.encoder.encode(t_observation)
                )
                action = t_action.cpu().numpy()

            observation, reward, terminated, truncated, info = self.eval_env.step(
                action
            )
            done = terminated or truncated
            self.eval_dataset.add(
                action=action,
                reward=reward,
                next_observation=observation,
                discount=1.0 - done,
                terminated=terminated,
                truncated=truncated,
            )

            step_metrics = {
                metric: value
                for metric, value in info.items()
                if metric in metrics.METRIC_AGGREGATION_FNS
            }
            step_metrics["reward"] = reward
            step_metrics["length"] = 1
            step_metrics_list.append(step_metrics)

        episode_metrics = metrics.aggregate_metrics(step_metrics_list)

        return episode_metrics
