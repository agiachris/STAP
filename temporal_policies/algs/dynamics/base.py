import abc
import pathlib
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym  # type: ignore
import numpy as np
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies.algs import base as algorithms
from temporal_policies.utils import logger, nest, trainer, timing, utils
from temporal_policies import datasets

PrimitiveConfig = Dict[str, Any]
TaskConfig = List[PrimitiveConfig]

Batch = nest.NestedStructure


def load_policies(
    task_config: TaskConfig, checkpoint_paths: List[str], device: str = "auto"
) -> List[algorithms.Algorithm]:
    """Loads the policy checkpoints with deterministic replay buffers to be used
    to train the dynamics model.

    Args:
        task_config: Ordered list of primitive (policy) configs.
        checkpoint_paths: Ordered list of policy checkpoints.
        device: Torch device.

    Returns:
        Ordered list of policies with loaded replay buffers.
    """
    policies: List[algorithms.Algorithm] = []
    for checkpoint_path in checkpoint_paths:
        policy = trainer.load_from_path(checkpoint_path, device=device, strict=True)
        policy.eval_mode()
        policy.setup_datasets()
        policies.append(policy)

    return policies


class DynamicsModel(abc.ABC):
    """Base dynamics class."""

    def __init__(
        self,
        policies: List[algorithms.Algorithm],
        network_class: Type[torch.nn.Module],
        network_kwargs: Dict[str, Any],
        dataset_class: Type[torch.utils.data.IterableDataset],
        dataset_kwargs: Dict[str, Any],
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]],
        scheduler_kwargs: Dict[str, Any],
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Dynamics model network class.
            network_kwargs: Kwargs for network class.
            dataset_class: Dynamics model dataset class.
            dataset_kwargs: Kwargs for dataset class.
            optimizer_class: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler_class: Dynamics model learning rate scheduler class.
            scheduler_class: Kwargs for scheduler class.
        """
        self._policies = policies
        self._loss = torch.nn.MSELoss()
        self._network = network_class(**network_kwargs).to(self.device)
        self._dataset, self._eval_dataset = _construct_datasets(
            policies, dataset_class, dataset_kwargs
        )
        self._optimizer = optimizer_class(self.network.parameters(), **optimizer_kwargs)
        if scheduler_class is not None:
            self._scheduler = scheduler_class(
                optimizer=self._optimizer, **scheduler_kwargs
            )
        else:
            self._scheduler = None
        self._steps = 0
        self._epochs = 0

    @property
    def policies(self) -> List[algorithms.Algorithm]:
        """Ordered list of policies used to perform the task."""
        return self._policies

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Training optimizer."""
        return self._optimizer

    @property
    def scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Learning rate scheduler."""
        return self._scheduler

    @property
    def network(self) -> torch.nn.Module:
        """Dynamics model network."""
        return self._network

    @property
    def dataset(self) -> torch.utils.data.IterableDataset:
        """Train dataset."""
        return self._dataset

    @property
    def eval_dataset(self) -> torch.utils.data.IterableDataset:
        """Eval dataset."""
        return self._eval_dataset

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._policies[0].device

    @property
    def steps(self) -> int:
        """Current training step."""
        return self._steps

    @property
    def epochs(self) -> int:
        """Current epoch."""
        return self._epochs

    def train_mode(self) -> None:
        """Switch to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Switch to eval mode."""
        self.network.eval()

    def predict(
        self,
        observation: Any,
        idx_policy: torch.Tensor,
        action: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predicts the next latent state given the current observation and
        action.

        Args:
            observation: Unencoded observation (common to all primitives).
            idx_policy: Index of executed policy.
            action: Policy action.

        Returns:
            Prediction of next latent state.
        """
        latent = self.encode(observation, idx_policy)
        latent_next = self.forward(latent, idx_policy, action)
        return latent_next

    def forward(
        self,
        latent: torch.Tensor,
        idx_policy: torch.Tensor,
        action: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predicts the next latent state given the current latent state and
        action.

        Args:
            latent: Current latent state.
            idx_policy: Index of executed policy.
            action: Policy action.

        Returns:
            Prediction of next latent state.
        """
        dz = self.network(latent, idx_policy, action)
        return latent + dz

    def encode(self, observation: Any, idx_policy: torch.Tensor) -> torch.Tensor:
        """Encodes the observation into a latent state vector.

        The base class returns the original observation.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.

        Returns:
            Encoded observation.
        """
        assert type(observation) is torch.Tensor
        return observation

    def compute_loss(
        self,
        observation: Any,
        idx_policy: torch.Tensor,
        action: List[torch.Tensor],
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the L2 loss between the predicted next latent and the latent
        encoded from the given next observation.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            action: Policy parameters.
            next_observation: Next observation.

        Returns:
            L2 loss.
        """
        # Predict next latent state.
        next_latent_pred = self.predict(observation, idx_policy, action)

        # Encode next latent state.
        next_latent = self.encode(next_observation, idx_policy)

        # Compute L2 loss.
        l2_loss = self._loss(next_latent_pred, next_latent)
        return l2_loss

    def save(self, path: Union[str, pathlib.Path], name: str) -> None:
        """Saves a checkpoint of the model and the optimizers.

        Args:
            path: Directory of checkpoint.
            name: Name of checkpoint (saved as `path/name.pt`).
        """
        save_dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(save_dict, pathlib.Path(path) / f"{name}.pt")

    def load(self, checkpoint: Union[str, pathlib.Path], strict: bool = True) -> None:
        """Loads the model from the given checkpoint.

        Args:
            checkpoint: Checkpoint path.
            strict: Strictly enforce matching state dict keys.
        """
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.network.load_state_dict(ckpt["network"], strict=strict)
        if strict:
            for key, val in self.optimizer.items():
                val.load_state_dict(ckpt["optim"][key])

    def train(
        self,
        path: str,
        total_steps: int,
        # schedule: bool = False,
        # schedule_kwargs: Dict = {},
        log_freq: int = 100,
        eval_freq: int = 1000,
        max_eval_steps: int = 100,
        workers: int = 4,
        profile_freq: int = -1,
    ) -> None:
        """Trains the dynamics model.

        Args:
            path: Output path.
            total_steps: Number of total training steps to perfrom across
                multiple calls to `train()`.
            log_freq: Logging frequency.
            eval_freq: Evaluation frequency.
            max_eval_steps: Maximum steps per evaluation.
            workers: Number of dataloader workers.
            profile_freq: Profiling frequency.
        """
        worker_init_fn = algorithms._worker_init_fn if workers > 0 else None
        pin_memory = self.device.type == "cuda"
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=pin_memory,
        )

        self.train_mode()

        log = logger.Logger(path=path)
        profiler = timing.Profiler(disabled=True)
        timer = timing.Timer()
        timer.tic("log_interval")

        train_losses = []
        min_eval_loss = float("inf")

        batches = iter(dataloader)
        for step in tqdm.tqdm(range(self.steps, total_steps)):
            self._steps = step

            # Configure profiler.
            if profile_freq > 0 and self.steps % profile_freq == 0:
                profiler.enable()
            else:
                profiler.disable()

            # Get next batch.
            profiler.tic("dataset")
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(dataloader)
                self._epochs += 1
                continue
            profiler.toc("dataset")

            # Train step.
            profiler.tic("train_step")
            loss = self._train_step(batch)
            train_losses.append(loss)
            profiler.toc("train_step")

            # Log.
            if self.steps % log_freq == 0:
                algorithms.log_from_dict(log, {"l2_loss": train_losses}, "train")
                algorithms.log_from_dict(log, profiler.collect_profiles(), "time")
                log.record("time/epochs", self.epochs)
                log.record(
                    "time/steps_per_second",
                    log_freq / timer.toc("log_interval", set_tic=True),
                )
                log.dump(step=self.steps)

            # Evaluate.
            if self.steps % eval_freq == 0:
                eval_loss = self._evaluate(eval_dataloader, max_eval_steps)

                # Save current model.
                log.record("eval/l2_loss", eval_loss)
                log.dump(step=self.steps, dump_csv=True)
                self.save(path, "final_model")

                # Save best model.
                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    self.save(path, "best_model")

    def _train_step(self, batch: Dict[str, Any]) -> int:
        """Executes one training step.

        Args:
            batch: Replay buffer batch.

        Returns:
            Computed loss.
        """
        self.optimizer.zero_grad()
        batch = self._format_batch(batch)
        loss = self.compute_loss(**batch)

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(loss)

        return loss.cpu().detach().numpy()

    def _evaluate(self, dataloader: torch.utils.data.DataLoader, max_steps: int) -> int:
        """Evaluates the model.

        Args:
            dataloader: Eval dataloader.
            max_steps: Maximum eval steps.

        Returns:
            Mean eval loss.
        """
        self.eval_mode()

        with torch.no_grad():
            # Evaluate on eval dataset.
            eval_losses = []
            for eval_step, batch in enumerate(tqdm.tqdm(dataloader)):
                if eval_step == max_steps:
                    break
                batch = self._format_batch(batch)
                loss = self.compute_loss(**batch)
                eval_losses.append(loss.cpu().numpy())

        self.train_mode()

        return np.mean(eval_losses)

    def _format_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the replay buffer batch for the dynamics model.

        Args:
            batch: Replay buffer batch.

        Returns:
            Dict with (observation, idx_policy, action, next_observation).
        """
        batch = {
            "observation": batch["observation"],
            "idx_policy": batch["action"]["idx_policy"],
            "action": batch["action"]["action"],
            "next_observation": batch["next_observation"],
        }

        # Convert items to tensor if they are not.
        if not utils.contains_tensors(batch):
            batch = utils.to_tensor(batch)
        batch = utils.to_device(batch, self.device)
        return batch


def _construct_datasets(
    policies: List[algorithms.Algorithm],
    dataset_class: Type[torch.utils.data.IterableDataset],
    dataset_kwargs: Dict[str, Any],
) -> Tuple[torch.utils.data.IterableDataset, torch.utils.data.IterableDataset]:
    """Constructs the dynamics model datasets from the policy replay buffers.

    The policy training datasets are used to create the dynamics model training
    dataset, and likewise for eval. The entries are not shuffled.

    Args:
        policies: Policies with loaded replay buffers.
        dataset_class: Output dataset class.
        dataset_kwargs: Kwargs for dataset class.

    Returns:
        (train_dataset, eval_dataset) 2-tuple.
    """

    def create_action_dict(
        action_space: gym.spaces.Dict, idx_policy: int, action: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Constructs an action dict for the given policy.

        Args:
            action_space: Action space for given policy.
            idx_policy: Policy index.
            action: Policy parameters.

        Returns:
            Action dict with nan padding.
        """
        action_padding = np.full(
            (action.shape[0], action_space["action"].shape[0] - action.shape[1]),
            float("nan"),
            dtype=action_space.dtype,
        )
        action_dict = {
            "idx_policy": np.full(action.shape[0], idx_policy, dtype=np.int32),
            "action": np.concatenate(
                (action.astype(action_space.dtype), action_padding), axis=1
            ),
        }
        return action_dict

    # Get observation space.
    assert all(
        policy.env.observation_space == policies[0].env.observation_space
        for policy in policies
    ), "Observation spaces must be the same among all policies."
    observation_space = policies[0].env.observation_space

    # Set the action space to the largest of the policy action spaces.
    len_action = max(policy.env.action_space.shape[0] for policy in policies)
    action_space = gym.spaces.Dict(
        {
            "idx_policy": gym.spaces.Discrete(len(policies)),
            "action": gym.spaces.Box(
                low=-1, high=1, shape=(len_action,), dtype=np.float32
            ),
        }
    )

    # Initialize the dynamics dataset.
    dataset = dataset_class(observation_space, action_space, **dataset_kwargs)
    dataset.initialize()
    eval_dataset = dataset_class(observation_space, action_space, **dataset_kwargs)
    eval_dataset.initialize()

    # Split dataset evenly among policies.
    num_entries_per_policy = dataset.capacity // len(policies)

    for idx_policy, policy in enumerate(policies):
        # Load policy replay buffers.
        policy.dataset.initialize()
        policy.eval_dataset.initialize()
        policy.dataset.load(max_entries=num_entries_per_policy)
        policy.eval_dataset.load(max_entries=num_entries_per_policy)

        # Load policy batch and reformat action.
        batch = dict(policy.dataset[:num_entries_per_policy])
        batch["action"] = create_action_dict(action_space, idx_policy, batch["action"])

        eval_batch = dict(policy.eval_dataset[:num_entries_per_policy])
        eval_batch["action"] = create_action_dict(
            action_space, idx_policy, eval_batch["action"]
        )

        dataset.add(batch=batch)
        eval_dataset.add(batch=eval_batch)

    return dataset, eval_dataset
