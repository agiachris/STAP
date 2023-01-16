import pathlib
from typing import Any, Dict, List, Mapping, Optional, Type, Union, Generator

import numpy as np
import torch
import tqdm

from temporal_policies import agents, datasets, processors
from temporal_policies.networks.encoders import IMAGE_ENCODERS
from temporal_policies.schedulers import DummyScheduler
from temporal_policies.trainers.base import Trainer
from temporal_policies.utils import configs, tensors
from temporal_policies.utils.typing import Batch, Scalar


class ValueTrainer(Trainer[agents.RLAgent, Batch, Batch]):
    """Value function trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        agent: agents.RLAgent,
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
        train_data_checkpoints: Optional[List[Union[str, pathlib.Path]]] = None,
        eval_data_checkpoints: Optional[List[Union[str, pathlib.Path]]] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        num_train_steps: int = 200000,
        num_eval_steps: int = 1000,
        eval_freq: int = 1000,
        checkpoint_freq: int = 10000,
        log_freq: int = 1000,
        profile_freq: Optional[int] = None,
        eval_metric: str = "q_loss",
        num_data_workers: int = 0,
        name: Optional[str] = None,
    ):
        """Prepares the agent trainer for training.

        Args:
            path: Training output path.
            agent: Agent to be trained.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            eval_dataset_kwargs: Kwargs for eval dataset.
            processor_class: Batch data processor class.
            processor_kwargs: Kwargs for processor.
            optimizer_class: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler_class: Optional optimizer scheduler class.
            scheduler_kwargs: Kwargs for scheduler class.
            train_data_checkpoints: Checkpoints to train data.
            eval_data_checkpoints: Checkpoints to validation data.
            checkpoint: Optional path to trainer checkpoint.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain.
            num_train_steps: Number of steps to train.
            num_eval_steps: Number of steps per evaluation.
            eval_freq: Evaluation frequency.
            checkpoint_freq: Checkpoint frequency.
            log_freq: Logging frequency.
            profile_freq: Profiling frequency.
            eval_metric: Metric to use for evaluation.
            num_data_workers: Number of workers to use for dataloader.
            name: Optional trainer name. Uses env name by default.
        """
        if name is None:
            name = agent.env.name
        path = pathlib.Path(path) / name

        if (train_data_checkpoints == checkpoint) or (eval_data_checkpoints == checkpoint):
            raise ValueError("Must provide either data checkpoints or ValueTrainer checkpoint.")

        # Load training dataset.
        dataset_class = configs.get_class(dataset_class, datasets)
        dataset_kwargs = dict(dataset_kwargs)
        dataset_kwargs["path"] = path / "train_data"
        dataset_kwargs["save_frequency"] = None
        dataset = dataset_class(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            **dataset_kwargs,
        )
        if train_data_checkpoints is not None:
            for train_data in train_data_checkpoints:
                dataset.load(train_data)

        # Load eval dataset.
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
        if eval_data_checkpoints is not None:
            for eval_data in eval_data_checkpoints:
                eval_dataset.load(eval_data)

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
            num_pretrain_steps=0,
            num_train_steps=num_train_steps,
            num_eval_steps=num_eval_steps,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            log_freq=log_freq,
            profile_freq=profile_freq,
            eval_metric=eval_metric,
            num_data_workers=num_data_workers,
        )

        if checkpoint is not None:
            self.load(checkpoint, strict=True)
        else:
            self.dataset.save()
            self.eval_dataset.save()

        self._eval_dataloader = self.create_dataloader(self.eval_dataset, self.num_data_workers)
        self._eval_batches = iter(self.eval_dataloader)

    @property
    def agent(self) -> agents.RLAgent:
        """Agent being trained."""
        return self.model

    @property
    def eval_dataloader(self) -> torch.utils.data.DataLoader:
        return self._eval_dataloader

    @property
    def eval_batches(self) -> Generator[Batch, None, None]:
        return self._eval_batches

    @property.setter
    def eval_batches(self, batches: Generator[Batch, None, None]):
        self._eval_batches = batches

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
            and isinstance(self.agent.encoder.network, IMAGE_ENCODERS)
        ):
            batch["observation"] = tensors.rgb_to_cnn(batch["observation"])  # type: ignore
            batch["next_observation"] = tensors.rgb_to_cnn(batch["next_observation"])  # type: ignore
        return super().process_batch(batch)

    def evaluate(self) -> List[Mapping[str, Union[Scalar, np.ndarray]]]:
        """Evaluates the model.

        Returns:
            Eval metrics.
        """
        self.eval_mode()

        with self.profiler.profile("evaluate"):
            metrics_list: List[Mapping[str, Union[Scalar, np.ndarray]]] = []
            pbar = tqdm.tqdm(
                range(self.num_eval_steps),
                desc=f"Eval {self.name}",
                dynamic_ncols=True,
            )
            for _ in pbar:
                
                try:
                    batch = next(self.eval_batches)
                except StopIteration:
                    self.eval_batches = iter(self.eval_dataloader)
                    batch = next(self.eval_batches)

                eval_metrics = self.agent.validation_step(batch)
                metrics_list.append(eval_metrics)
                pbar.set_postfix({self.eval_metric: eval_metrics[self.eval_metric]})

        self.train_mode()
        
        return metrics_list
