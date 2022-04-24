import pathlib
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies import agents, dynamics
from temporal_policies.trainers.agents import AgentTrainer
from temporal_policies.trainers.base import Trainer
from temporal_policies.trainers.dynamics import DynamicsTrainer
from temporal_policies.trainers.utils import load as load_trainer
from temporal_policies.utils import spaces, tensors


class UnifiedTrainer(Trainer):
    """Unified agents and dynamics trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        dynamics: dynamics.LatentDynamics,
        dynamics_trainer_config: Union[str, pathlib.Path, Dict[str, Any]],
        agent_trainer_configs: Sequence[Union[str, pathlib.Path, Dict[str, Any]]],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        num_pretrain_steps: int = 1000,
        num_train_steps: int = 100000,
        num_eval_steps: int = 100,
        eval_freq: int = 1000,
        log_freq: int = 100,
        profile_freq: Optional[int] = None,
    ):
        """Prepares the unified trainer for training.

        Args:
            path: Output path.
            dynamics: Dynamics model.
            dynamics_trainer_config: Dynamics trainer config.
            agent_trainer_configs: Agent trainer configs.
            checkpoint: Optional path to trainer checkpoint.
            device: Torch device.
            num_pretrain_steps: Number of steps to pretrain. Overrides
                agent trainer configs.
            num_train_steps: Number of steps to train. Overrides agent and
                dynamics trainer configs.
            num_eval_steps: Number of steps per evaluation. Overrides agent and
                dynamics trianer configs.
            eval_freq: Evaluation frequency. Overrides agent and dynamics
                trainer configs.
            log_freq: Logging frequency. Overrides agent and dynamics trainer
                configs.
            profile_freq: Profiling frequency. Overrides agent and dynamics
                trainer configs.
        """
        agent_trainer_kwargs = {
            "num_pretrain_steps": num_pretrain_steps,
            "num_train_steps": num_train_steps,
            "num_eval_episodes": num_eval_steps,
            "eval_freq": eval_freq,
            "log_freq": log_freq,
            "profile_freq": profile_freq,
        }
        dynamics_trainer_kwargs = {
            "num_train_steps": num_train_steps,
            "num_eval_steps": num_eval_steps,
            "eval_freq": eval_freq,
            "log_freq": log_freq,
            "profile_freq": profile_freq,
        }

        agent_trainers = []
        for agent_trainer_config, agent in zip(
            agent_trainer_configs, dynamics.policies
        ):
            assert isinstance(agent, agents.RLAgent)
            agent_trainer: AgentTrainer = load_trainer(  # type: ignore
                path=path,
                config=agent_trainer_config,
                agent=agent,
                device=device,
                **agent_trainer_kwargs,  # type: ignore
            )
            agent_trainers.append(agent_trainer)

        dynamics_trainer: DynamicsTrainer = load_trainer(  # type: ignore
            path=path,
            config=dynamics_trainer_config,
            dynamics=dynamics,
            agent_trainers=agent_trainers,
            device=device,
            **dynamics_trainer_kwargs,  # type: ignore
        )

        self._agent_trainers = agent_trainers
        self._dynamics_trainer = dynamics_trainer
        self._trainers: List[Trainer] = agent_trainers + [dynamics_trainer]  # type: ignore
        if len(set(trainer.name for trainer in self.trainers)) != len(self.trainers):
            trainer_names = [trainer.name for trainer in self.trainers]
            raise ValueError(f"All trainer names must be unique:\n{trainer_names}")

        self.num_pretrain_steps = num_pretrain_steps
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.profile_freq = profile_freq

        self._step = 0
        self._epoch = 0

    @property
    def agent_trainers(self) -> List[AgentTrainer]:
        """Agent trainers."""
        return self._agent_trainers

    @property
    def dynamics_trainer(self) -> DynamicsTrainer:
        """Dynamics trainer."""
        return self._dynamics_trainer

    @property
    def trainers(self) -> List[Trainer]:
        """Combined list of agent and dynamics trainers."""
        return self._trainers

    @property
    def agents(self) -> List[agents.RLAgent]:
        """Agent being trained."""
        return self.dynamics.policies  # type: ignore

    def increment_step(self):
        """Increments the training step for all trainers."""
        assert all(trainer.step == self.step for trainer in self.trainers)

        for trainer in self.trainers:
            trainer.increment_step()

        super().increment_step()

    def state_dict(self) -> Dict[str, Any]:
        """Gets the current trainer step/epoch."""
        state_dict: Dict[str, Any] = {
            "step": self.step,
            "epoch": self.epoch,
        }

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Sets the current trainer step/epoch."""
        self._step = state_dict["step"]
        self._epoch = state_dict["epoch"]

    def save(self, path: Union[str, pathlib.Path], name: str) -> pathlib.Path:
        """Saves checkpoints of all the trainers.

        Args:
            path: Directory of checkpoint.
            name: Name of checkpoint (saved as `path/name.pt`).
        """
        path = pathlib.Path(path)
        state_dict = self.state_dict()

        for trainer in self.trainers:
            state_dict[trainer.name] = trainer.save(path / trainer.name, name)

        checkpoint_path = path / f"{name}.pt"
        torch.save(state_dict, checkpoint_path)

        return checkpoint_path

    def load(self, checkpoint: Union[str, pathlib.Path], strict: bool = True) -> None:
        """Loads the trainer checkpoints to resume training.

        Args:
            checkpoint: Checkpoint path.
            strict: Make sure the state dict keys match.
        """
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)

        for trainer in self.trainers:
            trainer.load(state_dict[trainer.name], strict)

    def to(self, device: Union[str, torch.device]) -> Trainer:
        """Transfer networks to a device."""
        """Transfer networks to a device."""
        for trainer in self.trainers:
            trainer.to(device)
        return self

    def train_mode(self) -> None:
        """Switches to training mode."""
        for trainer in self.trainers:
            trainer.train_mode()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        for trainer in self.trainers:
            trainer.eval_mode()

    def collect_step(self, random: bool = False) -> Dict[str, Dict[str, float]]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics for each trainer.
        """
        collect_metrics = {}
        for agent_trainer in self.agent_trainers:
            collect_metrics[agent_trainer.name] = agent_trainer.collect_step(random)
        collect_metrics[self.dynamics_trainer.name] = {}

        return collect_metrics

    def pretrain(self) -> None:
        """Runs the pretrain phase for each agent."""
        for step in tqdm.tqdm(range(self.step, self.num_pretrain_steps)):
            collect_metrics = self.collect_step(random=True)
            self.increment_step()

    def profile_step(self) -> None:
        """Enables or disables profiling for the current step."""
        for trainer in self.trainers:
            trainer.profile_step()

    def train_step(  # type: ignore
        self, step: int, batch: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
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
            agent_batch: Dict[str, Any] = tensors.map_structure(  # type: ignore
                lambda x: x[idx_batch],
                {key: val for key, val in batch.items() if key != "idx_replay_buffer"},
            )
            agent_batch["action"] = spaces.subspace(
                agent_batch["action"], agent_trainer.agent.action_space
            )
            agent_train_metrics = agent_trainer.train_step(step, agent_batch)
            train_metrics[agent_trainer.name] = agent_train_metrics

        dynamics_train_metrics = self.dynamics_trainer.train_step(step, batch)
        train_metrics[self.dynamics_trainer.name] = dynamics_train_metrics

        return {
            key: {**collect_metrics[key], **train_metrics[key]} for key in train_metrics
        }

    def log_step(  # type: ignore
        self, metrics_list: Dict[str, List[Dict[str, float]]]
    ) -> Dict[str, List[Dict[str, float]]]:
        """Logs the metrics to Tensorboard if enabled for the current step.

        Args:
            metrics_list: List of metric dicts for each trainer accumulated
                since the last log_step.

        Returns:
            List of metric dicts for each trainer which haven't been logged yet.
        """
        metrics_list = {
            trainer.name: trainer.log_step(metrics_list[trainer.name])
            for trainer in self.trainers
        }
        return metrics_list

    def post_evaluate_step(  # type: ignore
        self, eval_metrics_list: Dict[str, List[Dict[str, float]]]
    ) -> None:
        """Logs the eval results and saves checkpoints.

        Args:
            eval_metrics_list: List of eval metric dicts for each trainer
                accumulated since the last post_evaluate_step.
        """
        for trainer in self.trainers:
            trainer.post_evaluate_step(eval_metrics_list[trainer.name])

    def train(self) -> None:
        """Trains the agents and dynamics together."""
        for trainer in self.trainers:
            trainer.dataset.initialize()
            trainer.eval_dataset.initialize()

        # Pretrain.
        self.pretrain()

        # Train.
        self.train_mode()
        metrics_list: Dict[str, List[Dict[str, float]]] = {
            trainer.name: [] for trainer in self.trainers
        }
        dataloader = self.dynamics_trainer.create_dataloader(
            self.dynamics_trainer.dataset, self.dynamics_trainer.num_data_workers
        )
        batches = iter(dataloader)
        for _ in tqdm.tqdm(
            range(self.step, self.num_pretrain_steps + self.num_train_steps)
        ):
            self.profile_step()

            # Get next batch.
            with self.dynamics_trainer.profiler.profile("dataset"):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(dataloader)
                    batch = next(batches)
                    self.increment_epoch()

            # Train step.
            train_metrics = self.train_step(self.step, batch)

            # Log.
            for key, train_metric in train_metrics.items():
                metrics_list[key].append(train_metric)
            metrics_list = self.log_step(metrics_list)

            # Evaluate.
            if self.step % self.eval_freq == 0:
                for trainer in self.trainers:
                    trainer.profiler.enable()
                eval_metrics_list = self.evaluate()
                self.post_evaluate_step(eval_metrics_list)

            self.increment_step()

    def evaluate(self) -> Dict[str, List[Dict[str, np.ndarray]]]:  # type: ignore
        """Evaluates the policies and dynamics model.

        Returns:
            Eval metrics for each trainer.
        """
        eval_metrics_list = {
            trainer.name: trainer.evaluate() for trainer in self.trainers
        }
        return eval_metrics_list
