import abc
import pathlib
import random
from typing import Any, Dict, Generic, List, Mapping, Optional, TypeVar, Union

import numpy as np
import torch
import tqdm

from temporal_policies import datasets, processors
from temporal_policies.utils import logging, metrics, tensors, timing
from temporal_policies.utils.typing import ModelType, Scalar


DatasetBatchType = TypeVar("DatasetBatchType", bound=Mapping)
ModelBatchType = TypeVar("ModelBatchType", bound=Mapping)


class Trainer(abc.ABC, Generic[ModelType, ModelBatchType, DatasetBatchType]):
    """Base trainer class."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        model: ModelType,
        dataset: datasets.ReplayBuffer,
        eval_dataset: datasets.ReplayBuffer,
        processor: processors.Processor,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
        checkpoint: Optional[Union[str, pathlib.Path]],
        device: str,
        num_pretrain_steps: int,
        num_train_steps: int,
        num_eval_steps: int,
        eval_freq: int,
        checkpoint_freq: int,
        log_freq: int,
        profile_freq: Optional[int],
        eval_metric: str,
        num_data_workers: int,
    ):
        """Prepares the trainer for training.

        Args:
            path: Training output path.
            model: Model to be trained.
            dataset: Train dataset.
            eval_dataset: Eval dataset.
            processor: Batch data processor.
            optimizers: Model optimizers.
            schedulers: Optimizer schedulers.
            checkpoint: Optional path to trainer checkpoint.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain.
            num_train_steps: Number of steps to train.
            num_eval_steps: Number of steps per evaluation.
            eval_freq: Evaluation frequency.
            checkpoint_freq: Checkpoint frequency (separate from latest/best
                eval checkpoints).
            log_freq: Logging frequency.
            profile_freq: Profiling frequency.
            eval_metric: Metric to use for evaluation.
            num_data_workers: Number of workers to use for dataloader.
        """
        self._path = pathlib.Path(path)
        self._model = model
        self._dataset = dataset
        self._eval_dataset = eval_dataset

        self._processor = processor
        self._optimizers = optimizers
        self._schedulers = schedulers

        self.num_pretrain_steps = num_pretrain_steps
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_freq = log_freq
        self.profile_freq = profile_freq
        self.num_data_workers = num_data_workers

        self.eval_metric = eval_metric
        self._best_eval_score = metrics.init_metric(self.eval_metric)

        self._log = logging.Logger(path=self.path)
        self._profiler = timing.Profiler(disabled=True)
        self._timer = timing.Timer()

        self._step = 0
        self._epoch = 0

        self.to(device)

        if checkpoint is not None:
            self.load(checkpoint, strict=True)

    @property
    def name(self) -> str:
        """Trainer name, equivalent to the last subdirectory in the path."""
        return self.path.name

    @property
    def model(self) -> ModelType:
        """Training model."""
        return self._model

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
    def best_eval_score(self) -> float:
        """Best eval score so far."""
        return self._best_eval_score

    @property
    def path(self) -> pathlib.Path:
        """Training output path."""
        return self._path

    @property
    def dataset(self) -> datasets.ReplayBuffer:
        """Train dataset."""
        return self._dataset

    @property
    def eval_dataset(self) -> datasets.ReplayBuffer:
        """Eval dataset."""
        return self._eval_dataset

    @property
    def processor(self) -> processors.Processor:
        """Data preprocessor."""
        return self._processor

    @property
    def optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Optimizers."""
        return self._optimizers

    @property
    def schedulers(
        self,
    ) -> Dict[str, torch.optim.lr_scheduler._LRScheduler]:
        """Learning rate schedulers."""
        return self._schedulers

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

    def state_dict(self) -> Dict[str, Any]:
        """Gets the trainer state dicts."""
        state_dict: Dict[str, Any] = {
            "step": self.step,
            "epoch": self.epoch,
            "best_eval_score": self.best_eval_score,
            "dataset_size": len(self.dataset),
            "eval_dataset_size": len(self.eval_dataset),
        }
        state_dict["optimizers"] = {
            key: optimizer.state_dict() for key, optimizer in self.optimizers.items()
        }
        if self.schedulers is not None:
            state_dict["schedulers"] = {
                key: scheduler.state_dict()
                for key, scheduler in self.schedulers.items()
            }
        state_dict.update(self.model.state_dict())
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Loads the trainer state dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self._step = state_dict["step"]
        self._epoch = state_dict["epoch"]
        self._best_eval_score = state_dict["best_eval_score"]
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][key])
        if self.schedulers is not None:
            for key, scheduler in self.schedulers.items():
                scheduler.load_state_dict(state_dict["schedulers"][key])
        self.model.load_state_dict(state_dict)

    def save(self, path: Union[str, pathlib.Path], name: str) -> pathlib.Path:
        """Saves a checkpoint of the trainer and model.

        Args:
            path: Directory of checkpoint.
            name: Name of checkpoint (saved as `path/name.pt`).
        """
        checkpoint_path = pathlib.Path(path) / f"{name}.pt"
        state_dict = self.state_dict()
        torch.save(state_dict, checkpoint_path)

        return checkpoint_path

    def load(
        self,
        checkpoint: Union[str, pathlib.Path],
        strict: bool = True,
        dataset_size: Optional[int] = None,
        eval_dataset_size: Optional[int] = None,
    ) -> None:
        """Loads the trainer checkpoint to resume training.

        Args:
            checkpoint: Checkpoint path.
            strict: Make sure the state dict keys match.
        """
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)

        path = pathlib.Path(checkpoint).parent
        self.dataset.path = path / "train_data"
        self.eval_dataset.path = path / "eval_data"
        # TODO: Uncomment this to load the replay_buffer at the time of saving.
        # if dataset_size is None:
        #     dataset_size = state_dict["dataset_size"]
        # if eval_dataset_size is None:
        #     eval_dataset_size = state_dict["eval_dataset_size"]
        self.dataset.load(max_entries=dataset_size)
        self.eval_dataset.load(max_entries=eval_dataset_size)

    def to(self, device: Union[str, torch.device]) -> "Trainer":
        """Transfer networks to a device."""
        self._device = tensors.device(device)
        self.processor.to(self.device)
        self.model.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches to training mode."""
        self.processor.train()
        self.model.train_mode()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        self.processor.eval()
        self.model.eval_mode()

    def process_batch(self, batch: DatasetBatchType) -> ModelBatchType:
        """Processes replay buffer batch for training.

        Args:
            batch: Replay buffer batch.

        Returns:
            Processed batch.
        """
        return tensors.to(batch, self.device)

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

    def get_lr_metrics(self) -> Dict[str, float]:
        """Gets learning rate metrics for logging."""
        return {
            key: scheduler.get_last_lr()[0]
            for key, scheduler in self.schedulers.items()
        }

    def profile_step(self) -> None:
        """Enables or disables profiling for the current step."""
        if self.profile_freq is not None and self.step % self.profile_freq == 0:
            self.profiler.enable()
        else:
            self.profiler.disable()

    def train_step(self, step: int, batch: DatasetBatchType) -> Mapping[str, float]:
        """Performs a single training step.

        Args:
            step: Training step.
            batch: Training batch.

        Returns:
            Dict of training metrics for logging.
        """
        with self.profiler.profile("train"):
            model_batch = self.process_batch(batch)
            return self.model.train_step(
                step, model_batch, self.optimizers, self.schedulers
            )

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
        self.log.log("lr", self.get_lr_metrics())
        self.log.flush(step=self.step)

        return []

    def post_evaluate_step(
        self, eval_metrics_list: List[Mapping[str, Union[Scalar, np.ndarray]]]
    ) -> None:
        """Logs the eval results and saves checkpoints.

        Args:
            eval_metrics_list: List of eval metric dicts accumulated since the
                last post_evaluate_step.
        """
        if not eval_metrics_list:
            return

        eval_metrics = metrics.collect_metrics(eval_metrics_list)
        self.log.log("eval", eval_metrics)
        self.log.flush(step=self.step, dump_csv=True)
        self.save(self.path, "final_trainer")
        self.model.save(self.path, "final_model")

        # Save best model.
        eval_score = np.mean(eval_metrics[self.eval_metric])
        is_eval_better = (
            metrics.best_metric(self.eval_metric, eval_score, self.best_eval_score)
            == eval_score
        )
        if is_eval_better:
            self._best_eval_score = eval_score
            self.save(self.path, "best_trainer")
            self.model.save(self.path, "best_model")

    def create_dataloader(
        self, dataset: torch.utils.data.IterableDataset, workers: int = 0
    ) -> torch.utils.data.DataLoader:
        """Creates a Torch dataloader for the given dataset.

        Args:
            dataset: Iterable dataset.
            workers: Number of data loader workers.

        Returns:
            Torch dataloader.
        """

        def _worker_init_fn(worker_id: int) -> None:
            random_state = np.random.get_state()
            assert isinstance(random_state, tuple)
            seed = random_state[1][0] + worker_id
            np.random.seed(seed)
            random.seed(seed)

        worker_init_fn = None if workers == 0 else _worker_init_fn
        pin_memory = self.device.type == "cuda"
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
        )
        return dataloader

    def pretrain(self) -> None:
        """Runs the pretrain phase."""
        return

    def train(self) -> None:
        """Trains the model."""
        dataloader = self.create_dataloader(self.dataset, self.num_data_workers)
        self.eval_dataset.initialize()

        # Pretrain.
        self.pretrain()
        assert self.step >= self.num_pretrain_steps

        # Evaluate.
        self.profiler.enable()
        eval_metrics_list = self.evaluate()
        self.post_evaluate_step(eval_metrics_list)

        # Train.
        self.train_mode()
        metrics_list = []
        batches = iter(dataloader)
        pbar = tqdm.tqdm(
            range(self.step - self.num_pretrain_steps, self.num_train_steps),
            desc=f"Train {self.name}",
            dynamic_ncols=True,
        )
        for train_step in pbar:
            self.profile_step()

            # Get next batch.
            with self.profiler.profile("dataset"):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(dataloader)
                    batch = next(batches)
                    self.increment_epoch()

            # Train step.
            train_metrics = self.train_step(self.step, batch)
            try:
                pbar.set_postfix({self.eval_metric: train_metrics[self.eval_metric]})
            except KeyError:
                pass

            # Log.
            metrics_list.append(train_metrics)
            metrics_list = self.log_step(metrics_list)

            self.increment_step()
            eval_step = train_step + 1

            # Evaluate.
            if eval_step % self.eval_freq == 0:
                self.profiler.enable()
                eval_metrics_list = self.evaluate()
                self.post_evaluate_step(eval_metrics_list)

            # Checkpoint.
            if eval_step % self.checkpoint_freq == 0:
                self.save(self.path, f"ckpt_trainer_{eval_step}")
                self.model.save(self.path, f"ckpt_model_{eval_step}")

    @abc.abstractmethod
    def evaluate(self) -> List[Mapping[str, Union[Scalar, np.ndarray]]]:
        """Evaluates the model.

        Returns:
            Eval metrics.
        """
        pass
