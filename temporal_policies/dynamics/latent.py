import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies import agents, datasets, networks
from temporal_policies.dynamics import base as dynamics
from temporal_policies.utils import configs, logger, spaces, tensors, timing


class LatentDynamics(dynamics.Dynamics):
    """Base dynamics class."""

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[torch.nn.Module]],
        network_kwargs: Dict[str, Any],
        dataset_class: Union[str, Type[torch.utils.data.IterableDataset]],
        dataset_kwargs: Dict[str, Any],
        optimizer_class: Union[str, Type[torch.optim.Optimizer]],
        optimizer_kwargs: Dict[str, Any],
        scheduler_class: Optional[
            Union[str, Type[torch.optim.lr_scheduler._LRScheduler]]
        ] = None,
        scheduler_kwargs: Dict[str, Any] = {},
        state_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Dynamics model network class.
            network_kwargs: Kwargs for network class.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            optimizer_class: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler_class: Optional dynamics model learning rate scheduler class.
            scheduler_kwargs: Kwargs for scheduler class.
            state_space: Optional state space.
            action_space: Optional action space.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        network_class = configs.get_class(network_class, networks)
        self._network = network_class(**network_kwargs)

        super().__init__(
            policies=policies,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        self._loss = torch.nn.MSELoss()

        self._dataset, self._eval_dataset = _construct_datasets(
            policies, dataset_class, dataset_kwargs
        )

        optimizer_class = configs.get_class(optimizer_class, torch.optim)
        self._optimizer = optimizer_class(self.network.parameters(), **optimizer_kwargs)

        if scheduler_class is not None:
            scheduler_class = configs.get_class(
                scheduler_class, torch.optim.lr_scheduler
            )
            self._scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)
        else:
            self._scheduler = None

        self._steps = 0
        self._epochs = 0

        if checkpoint is not None:
            self.load(checkpoint, strict=False)

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
    def steps(self) -> int:
        """Current training step."""
        return self._steps

    @property
    def epochs(self) -> int:
        """Current epoch."""
        return self._epochs

    def to(self, device: Union[str, torch.device]) -> dynamics.Dynamics:
        """Transfers networks to device."""
        super().to(device)
        self.network.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switch to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Switch to eval mode."""
        self.network.eval()

    def forward(
        self,
        state: torch.Tensor,
        idx_policy: torch.Tensor,
        action: Sequence[torch.Tensor],
        policy_args: Optional[Any] = None,
    ) -> torch.Tensor:
        """Predicts the next latent state given the current latent state and
        action.

        Args:
            state: Current latent state.
            idx_policy: Index of executed policy.
            action: Policy action.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next latent state.
        """
        dz = self.network(state, idx_policy, action)
        return state + dz

    def compute_loss(
        self,
        observation: Any,
        idx_policy: torch.Tensor,
        action: Sequence[torch.Tensor],
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
        latent = self.encode(observation, idx_policy)
        next_latent_pred = self.forward(latent, idx_policy, action)

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
        worker_init_fn = agents.rl._worker_init_fn if workers > 0 else None
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
                agents.rl.log_from_dict(log, {"l2_loss": train_losses}, "train")
                agents.rl.log_from_dict(log, profiler.collect_profiles(), "time")
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

        return tensors.to(batch, self.device)  # type: ignore


def _construct_datasets(
    policies: Sequence[agents.RLAgent],
    dataset_class: Union[str, Type[torch.utils.data.IterableDataset]],
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
        policy.observation_space == policies[0].observation_space for policy in policies
    ), "Observation spaces must be the same among all policies."
    observation_space = policies[0].observation_space

    # Set the action space to the largest of the policy action spaces.
    action_space = gym.spaces.Dict(
        {
            "idx_policy": gym.spaces.Discrete(len(policies)),
            "action": spaces.overlay_boxes(
                [policy.action_space for policy in policies]
            ),
        }
    )

    # Initialize the dynamics dataset.
    if isinstance(dataset_class, str):
        dataset_class = configs.get_class(dataset_class, datasets)
    dataset = dataset_class(observation_space, action_space, **dataset_kwargs)
    dataset.initialize()
    eval_dataset = dataset_class(observation_space, action_space, **dataset_kwargs)
    eval_dataset.initialize()

    # Split dataset evenly among policies.
    num_entries_per_policy = dataset.capacity // len(policies)

    for idx_policy, policy in enumerate(policies):
        # Load policy replay buffers.
        policy.setup_datasets()
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
