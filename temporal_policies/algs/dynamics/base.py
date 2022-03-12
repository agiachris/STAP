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

        # Load replay buffers.
        policy.setup_datasets()
        replay_buffer_path = pathlib.Path(checkpoint_path).parent / "replay_buffers"
        policy.dataset.preload_path = replay_buffer_path / "train"
        policy.eval_dataset.preload_path = replay_buffer_path / "eval"

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
        """
        self._policies = policies
        self._loss = torch.nn.MSELoss()
        self._network = network_class(**network_kwargs).to(self.device)
        self._dataset, self._eval_dataset = _construct_datasets(
            policies, dataset_class, dataset_kwargs
        )
        self._optimizer = optimizer_class(self.network.parameters(), **optimizer_kwargs)
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
        policy_params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predicts the next latent state given the current observation and
        action.

        Args:
            observation: Unencoded observation (common to all primitives).
            idx_policy: Index of executed policy.
            policy_params: Policy parameters.

        Returns:
            Prediction of next latent state.
        """
        latent = self.encode(observation, idx_policy)
        latent_next = self.forward(latent, idx_policy, policy_params)
        return latent_next

    def forward(
        self,
        latent: torch.Tensor,
        idx_policy: torch.Tensor,
        policy_params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Predicts the next latent state given the current latent state and
        action.

        Args:
            latent: Current latent state.
            idx_policy: Index of executed policy.
            policy_params: Policy parameters.

        Returns:
            Prediction of next latent state.
        """
        dz = self.network(latent, idx_policy, policy_params)
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
        policy_params: List[torch.Tensor],
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the L2 loss between the predicted next latent and the latent
        encoded from the given next observation.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_params: Policy parameters.
            next_observation: Next observation.

        Returns:
            L2 loss.
        """
        # Predict next latent state.
        next_latent_pred = self.predict(observation, idx_policy, policy_params)

        # Encode next latent state.
        next_latent = self.encode(next_observation, idx_policy)

        # Compute L2 loss.
        l2_loss = self._loss(next_latent_pred, next_latent)
        return l2_loss

    def save(self, path: str, name: str) -> None:
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

    def load(self, checkpoint: str, strict: bool = True) -> None:
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
            # shuffle=True,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
            # collate_fn=self.collate_fn,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=None,
            # shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            # collate_fn=self.collate_fn,
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
                break
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
        batch = self._format_batch(batch)
        loss = self.compute_loss(**batch)
        loss.backward()
        self.optimizer.step()

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
            Dict with (observation, idx_policy, policy_params, next_observation).
        """
        # Collect policy params.
        idx_policy = batch["action"]["idx_policy"]
        policy_params = [
            batch["action"][f"policy_{i}"][idx_batch]
            for idx_batch, i in enumerate(idx_policy)
        ]
        batch = {
            "observation": batch["obs"],
            "idx_policy": idx_policy,
            "policy_params": policy_params,
            "next_observation": batch["next_obs"],
        }

        # Convert items to tensor if they are not.
        if not utils.contains_tensors(batch):
            batch = utils.to_tensor(batch)
        batch = utils.to_device(batch, self.device)
        return batch


def _add_to_replay_buffer(
    replay_buffer: datasets.ReplayBuffer,
    source: Union[Batch, datasets.ReplayBuffer],
    max_entries: int = sys.maxsize,
) -> int:
    """Adds entries from the source to the given replay buffer.

    Args:
        replay_buffer: Replay buffer.
        source: Batch or another replay buffer.
        max_entries: Maximum number of entries to add.

    Returns:
        Number of entries added.
    """
    def to_batch(
        replay_buffer: datasets.ReplayBuffer, trim: bool = True
    ) -> Batch:
        batch = {
            "obs": replay_buffer._obs_buffer,
            "action": replay_buffer._action_buffer,
            "reward": replay_buffer._reward_buffer,
            "done": replay_buffer._done_buffer,
            "discount": replay_buffer._discount_buffer,
        }
        if not trim:
            return batch

        idx_end = replay_buffer._size if trim else replay_buffer.capacity
        return nest.map_structure(lambda x: x[:idx_end], batch)

    def add_to_buffer(
        dest: np.ndarray, src: np.ndarray, idx: int, max_entries: int
    ) -> int:
        assert max_entries >= 0

        len_src = min(max_entries, len(src))
        idx_end = min(len(dest), idx + len_src)
        num_added = idx_end - idx
        dest[idx:idx_end] = src[:num_added]

        if num_added < len_src:
            return num_added + add_to_buffer(
                dest, src[num_added:], 0, max_entries - num_added
            )
        else:
            return num_added

    def add_to_batch(
        dest: Batch, src: Batch, idx: int, max_entries: int
    ) -> int:
        num_added = nest.map_structure(
            lambda x, y: add_to_buffer(x, y, idx, max_entries), dest, src
        )
        return next(nest.structure_iterator(num_added))  # type: ignore

    dest = to_batch(replay_buffer, trim=False)
    src = to_batch(source) if isinstance(source, datasets.ReplayBuffer) else source
    num_added = add_to_batch(dest, src, replay_buffer._size, max_entries)

    replay_buffer._idx = (replay_buffer._size + num_added) % replay_buffer.capacity
    replay_buffer._size = min(replay_buffer.capacity, replay_buffer._size + num_added)

    return num_added


def _load_replay_buffer_from_disk(
    replay_buffer: datasets.ReplayBuffer,
    path: pathlib.Path,
    max_entries: int = sys.maxsize,
    transform_batch: Optional[Callable[[Batch], Batch]] = None,
) -> int:
    """Adds episodes stored on disk to the given replay buffer.

    Args:
        replay_buffer: Replay buffer.
        path: Directory where episodes are stored.
        max_entries: Maximum number of entries to add.
        transform_batch: Optional function to transform the batch from disk
            before adding to the replay buffer.

    Returns:
        Number of entries added.
    """
    # Sort episodes by creation time/episode number.
    episode_paths = sorted(
        path.iterdir(), key=lambda f: tuple(map(int, re.split(r"T|_", f.stem)[:-1]))
    )
    num_added_total = 0
    for episode_path in tqdm.tqdm(episode_paths):
        with open(episode_path, "rb") as f:
            batch: Batch = dict(np.load(f))

        if transform_batch is not None:
            batch = transform_batch(batch)

        num_added = _add_to_replay_buffer(replay_buffer, batch, max_entries)
        num_added_total += num_added
        max_entries -= num_added
        if max_entries <= 0:
            break

    return num_added_total


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
            Action dict with null entries for other policies.
        """
        # TODO: Need better way to combine actions.
        action_dict = {
            "idx_policy": np.full(action.shape[0], idx_policy, dtype=np.int32),
            f"policy_{idx_policy}": action,
        }
        for key, space in action_space.items():
            if key in action_dict:
                continue
            action_dict[key] = np.full(
                (action.shape[0], *space.shape), float("nan"), dtype=np.float32
            )
        return action_dict

    assert all(
        policy.env.observation_space == policies[0].env.observation_space
        for policy in policies
    ), "Observation spaces must be the same among all policies."
    observation_space = policies[0].env.observation_space

    action_dict = {
        f"policy_{i}": policy.env.action_space for i, policy in enumerate(policies)
    }
    action_dict["idx_policy"] = gym.spaces.Discrete(len(policies))
    action_space = gym.spaces.Dict(action_dict)

    dataset = dataset_class(observation_space, action_space, **dataset_kwargs)
    eval_dataset = dataset_class(observation_space, action_space, **dataset_kwargs)

    for idx_policy, policy in enumerate(policies):

        def transform_batch(batch: Batch) -> Batch:
            batch["action"] = create_action_dict(  # type: ignore
                action_space, idx_policy, batch["action"]  # type: ignore
            )
            return batch

        _load_replay_buffer_from_disk(
            dataset,
            policy.dataset.preload_path,
            max_entries=dataset.capacity // len(policies),
            transform_batch=transform_batch,
        )
        _load_replay_buffer_from_disk(
            eval_dataset,
            policy.eval_dataset.preload_path,
            max_entries=eval_dataset.capacity // len(policies),
            transform_batch=transform_batch,
        )

    return dataset, eval_dataset
