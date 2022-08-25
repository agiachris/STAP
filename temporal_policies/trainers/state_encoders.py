import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import torch
import tqdm

from temporal_policies import datasets, encoders, processors
from temporal_policies.schedulers import DummyScheduler
from temporal_policies.trainers.agents import AgentTrainer
from temporal_policies.trainers.base import Trainer
from temporal_policies.trainers.utils import load as load_trainer
from temporal_policies.utils import configs, tensors
from temporal_policies.utils.typing import Scalar, StateBatch, StateEncoderBatch


# TODO: Not fully implemented


class StateEncoderTrainer(
    Trainer[encoders.StateEncoder, StateEncoderBatch, StateBatch]
):
    """State encoder trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        encoder: encoders.StateEncoder,
        dataset_class: Union[str, Type[datasets.StateBuffer]],
        dataset_kwargs: Dict[str, Any],
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
        policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        agent_trainers: Optional[Sequence[AgentTrainer]] = None,
        device: str = "auto",
        num_train_steps: int = 100000,
        num_eval_steps: int = 100,
        eval_freq: int = 1000,
        checkpoint_freq: int = 10000,
        log_freq: int = 100,
        profile_freq: Optional[int] = None,
        eval_metric: str = "loss",
        num_data_workers: int = 0,
    ):
        """Prepares the state encoder trainer for training.

        Args:
            path: Output path.
            encoder: Encoder model to train.
            dataset_class: State encoder model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            eval_dataset_kwargs: Kwargs for eval dataset.
            processor_class: Batch data processor calss.
            processor_kwargs: Kwargs for processor.
            optimizer_class: State encoder model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler_class: Optional optimizer scheduler class.
            scheduler_kwargs: Kwargs for scheduler class.
            checkpoint: Optional path to trainer checkpoint.
            policy_checkpoints: List of policy checkpoints. Either this or
                agent_trainers must be specified.
            agent_trainers: List of agent trainers. Either this or
                policy_checkpoints must be specified.
            device: Torch device.
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
        path = pathlib.Path(path) / "encoder"

        # Get agent trainers.
        if agent_trainers is None:
            if policy_checkpoints is None:
                raise ValueError(
                    "One of agent_trainers or policy_checkpoints must be specified"
                )
            agent_trainers = []
            for policy_checkpoint in policy_checkpoints:
                policy_checkpoint = pathlib.Path(policy_checkpoint)
                if policy_checkpoint.is_file():
                    trainer_checkpoint = (
                        policy_checkpoint.parent
                        / policy_checkpoint.name.replace("model", "trainer")
                    )
                else:
                    trainer_checkpoint = policy_checkpoint / "final_trainer.pt"
                agent_trainer = load_trainer(checkpoint=trainer_checkpoint)
                assert isinstance(agent_trainer, AgentTrainer)
                agent_trainers.append(agent_trainer)

        dataset_class = configs.get_class(dataset_class, datasets)
        dataset = dataset_class(**dataset_kwargs)
        eval_dataset = dataset_class(**dataset_kwargs)

        processor_class = configs.get_class(processor_class, processors)
        processor = processor_class(
            encoder.observation_space,
            **processor_kwargs,
        )

        optimizer_class = configs.get_class(optimizer_class, torch.optim)
        optimizers = encoder.create_optimizers(optimizer_class, optimizer_kwargs)

        scheduler_class = configs.get_class(scheduler_class, torch.optim.lr_scheduler)
        schedulers = {
            key: scheduler_class(optimizer, **scheduler_kwargs)
            for key, optimizer in optimizers.items()
        }

        if isinstance(encoder, encoders.VAE):
            encoder.train_setup(dataset)

        super().__init__(
            path=path,
            model=encoder,
            dataset=dataset,
            eval_dataset=eval_dataset,
            processor=processor,
            optimizers=optimizers,
            schedulers=schedulers,
            checkpoint=checkpoint,
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

        self._eval_dataloader: Optional[torch.utils.data.DataLoader] = None

    @property
    def encoder(self) -> encoders.StateEncoder:
        """State encoder being trained."""
        return self.model

    def process_batch(self, batch: StateBatch) -> StateEncoderBatch:
        """Formats the replay buffer batch for the state encoder model.

        Args:
            batch: Replay buffer batch.

        Returns:
            Dict with (observation, idx_policy, action, next_observation).
        """
        encoder_batch = StateEncoderBatch(
            observation=batch["observation"], state=batch["state"]
        )
        return tensors.to(encoder_batch, self.device)

    def evaluate(self) -> List[Mapping[str, Union[Scalar, np.ndarray]]]:
        """Evaluates the model.

        Returns:
            Eval metrics.
        """
        if self._eval_dataloader is None:
            self._eval_dataloader = self.create_dataloader(self.eval_dataset, 1)

        self.eval_mode()

        eval_metrics_list: List[Mapping[str, Union[Scalar, np.ndarray]]] = []
        pbar = tqdm.tqdm(
            self._eval_dataloader,
            desc=f"Eval {self.name}",
            dynamic_ncols=True,
        )
        for eval_step, batch in enumerate(pbar):
            if eval_step == self.num_eval_steps:
                break

            with torch.no_grad():
                batch = self.process_batch(batch)
                loss, eval_metrics = self.encoder.compute_loss(**batch)

            pbar.set_postfix({self.eval_metric: eval_metrics[self.eval_metric]})
            eval_metrics_list.append(eval_metrics)

        self.train_mode()

        return eval_metrics_list
