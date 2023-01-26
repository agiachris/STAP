import pathlib
from typing import Any, Dict, Mapping, Optional, Type, Union, List

import torch

from temporal_policies import agents, datasets, envs
from temporal_policies.utils import logging, timing, configs, tensors, metrics


COLLECTION_STRATEGIES = ["uniform", "balanced"]


class PrimitiveDatasetGenerator:
    """Primitive dataset generation script."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        agent: agents.RLAgent,
        dataset_size: int = 50000,
        collection_strategy: str = "uniform",
        min_success_ratio: Optional[float] = None,
        max_failure_ratio: Optional[float] = None,
        simulate: bool = True,
        dataset_class: Union[str, Type[datasets.ReplayBuffer]] = datasets.ReplayBuffer,
        dataset_kwargs: Dict[str, Any] = {},
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        env_kwargs: Dict[str, Any] = {},
        device: str = "auto",
        log_freq: int = 100,
        profile_freq: Optional[int] = None,
        num_data_workers: int = 0,
        name: Optional[str] = None,
    ):
        """Prepares the agent trainer for training.

        Args:
            path: Training output path.
            agent: Agent to be trained.
            dataset_size: Number of transitions to collect.
            collection_strategy: Data collection strategy.
            min_success_ratio: Minimum number of successful episodes in the buffer.
            max_failure_ratio: Maximum number of failed episodes in the buffer.
            simulate: If True, the sampled action is simulated in the environment.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            checkpoint: Optional path to trainer checkpoint.
            env_kwargs: Optional kwargs passed to EnvFactory.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain.
            log_freq: Logging frequency.
            profile_freq: Profiling frequency.
            num_data_workers: Number of workers to use for dataloader.
            name: Optional trainer name. Uses env name by default.
        """        
        if name is None:
            name = agent.env.name
        path = pathlib.Path(path) / name
        self._path = path

        # Dataset parameters.
        assert collection_strategy in COLLECTION_STRATEGIES
        if collection_strategy == "balanced":
            assert not (min_success_ratio is None and max_failure_ratio is None)
            min_success_ratio = 0.0 if min_success_ratio is None else min_success_ratio
            max_failure_ratio = 1.0 if max_failure_ratio is None else max_failure_ratio
        self._dataset_size = dataset_size
        self._collection_strategy = collection_strategy
        self._min_success_ratio = min_success_ratio
        self._max_failure_ratio = max_failure_ratio 
        self._simulate = simulate

        # Trainer parameters.
        self._agent = agent
        dataset_class = configs.get_class(dataset_class, datasets)
        dataset_kwargs = dict(dataset_kwargs)
        dataset_kwargs["path"] = path / "train_data"
        self._dataset = dataset_class(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            **dataset_kwargs,
        )

        self.log_freq = log_freq
        self.profile_freq = profile_freq
        self.num_data_workers = num_data_workers
        self._log = logging.Logger(path=self.path)
        self._profiler = timing.Profiler(disabled=True)
        self._timer = timing.Timer()
        self._device = tensors.device(device)

        self._num_failure = 0
        self._step = 0
        self._epoch = 0

    @property
    def name(self) -> str:
        """Trainer name, equivalent to the last subdirectory in the path."""
        return self.path.name

    @property
    def path(self) -> pathlib.Path:
        """Training output path."""
        return self._path

    @property
    def agent(self) -> agents.RLAgent:
        """Agent being trained."""
        return self._agent
    
    @property
    def env(self) -> envs.Env:
        """Agent env."""
        return self.agent.env
    
    @property
    def dataset(self) -> datasets.ReplayBuffer:
        """Train dataset."""
        return self._dataset

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    def increment_step(self):
        """Increments the training step."""
        self._step += 1

    @property
    def epoch(self) -> int:
        """Current training epochs."""
        return self._epoch

    def increment_epoch(self):
        """Increments the training epoch."""
        self._epoch += 1

    @property
    def log(self) -> logging.Logger:
        """Tensorboard logger."""
        return self._log

    @property
    def profiler(self) -> timing.Profiler:
        """Code profiler."""
        return self._profiler

    @property
    def timer(self) -> timing.Timer:
        """Code timer."""
        return self._timer

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device
    
    def train(self) -> None:
        """Trains the model."""
        self.dataset.initialize()
        
        log_freq = self.log_freq
        self.log_freq = min(log_freq, self._dataset_size // 10)

        metrics_list = []
        while self.step < self._dataset_size:
            collect_metrics = self.collect_step()
            if not self._simulate:
                self.increment_step()
            elif self._simulate and collect_metrics:
                metrics_list.append(collect_metrics)
                metrics_list = self.log_step(metrics_list, stage="collection")
                self.increment_step()

        self.log_freq = log_freq
        self.dataset.save()

    def collect_step(self) -> Mapping[str, Any]:
        """Collects data for the replay buffer.

        Returns:
            Collect metrics.
        """
        collect_metrics = {}
        with self.profiler.profile("collect"):
            observation, _ = self.env.reset()
            action = self.env.get_primitive().sample()

            if self._simulate:
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                discount = 1.0 - float(done)
                try:
                    policy_args = info["policy_args"]
                except KeyError:
                    policy_args = None

                assert done, "Must be in single-step environment."
                
                if self._collection_strategy == "balanced":
                    exceeding_max_num_failure = self._num_failure >= self._dataset_size * self._max_failure_ratio
                    violating_min_num_success = self._dataset_size - self._num_failure <= self._dataset_size * self._min_success_ratio
                    if reward == 0.0 and (exceeding_max_num_failure or violating_min_num_success):
                        return collect_metrics
                elif self._collection_strategy == "uniform":
                    pass
                else:
                    raise ValueError(f"Collection strategy {self._collection_strategy} is not supported.")

                self._num_failure += int(reward == 0.0)

                self.dataset.add(observation=observation)
                self.dataset.add(
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    discount=discount,
                    terminated=terminated,
                    truncated=truncated,
                    policy_args=policy_args,
                )
                collect_metrics.update({"reward": reward, "episode": self.epoch})
    
            else:
                self.dataset.add(observation=observation)
                self.dataset.add(
                    action=action,
                    reward=0.0,
                    next_observation=observation.copy(),
                    discount=0.0,
                    terminated=True,
                    truncated=True,
                    policy_args={},
                )

            return collect_metrics
    

    def log_step(
        self, metrics_list: List[Mapping[str, float]], stage: str = "train"
    ) -> List[Mapping[str, float]]:
        """Logs the metrics to Tensorboard if enabled for the current step.

        Args:
            metrics_list: List of metric dicts accumulated since the last
                log_step.
            stage: "train" or "pretrain".

        Returns:
            List of metric dicts which haven't been logged yet.
        """
        if self.step % self.log_freq != 0 or not metrics_list:
            return metrics_list

        log_metrics = metrics.collect_metrics(metrics_list)
        self.log.log(stage, log_metrics)
        self.log.log("time", self.get_time_metrics())
        self.log.flush(step=self.step)

        return []

    def get_time_metrics(self) -> Dict[str, float]:
        """Gets time metrics for logging."""
        time_metrics = self.profiler.collect_profiles()
        time_metrics["epoch"] = self.epoch
        if "log_interval" in self.timer.keys():
            time_metrics["steps_per_second"] = self.log_freq / self.timer.toc(
                "log_interval", set_tic=True
            )
        else:
            self.timer.tic("log_interval")
        return time_metrics

