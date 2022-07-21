import pathlib
from typing import Any, Dict, Generic, Optional, Union, Type

import numpy as np  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies import datasets, scod, envs, agents
from temporal_policies.networks.encoders import OracleEncoder, NormalizeObservation
from temporal_policies.trainers.agents import AgentTrainer
from temporal_policies.trainers.utils import load as load_trainer
from temporal_policies.utils import configs, tensors, logging, metrics
from temporal_policies.utils.typing import ObsType


class SCODTrainer(Generic[ObsType]):
    """SCOD trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        scod: scod.SCOD,
        dataset_class: Union[
            str, Type[torch.utils.data.IterableDataset]
        ] = datasets.ReplayBuffer,
        dataset_kwargs: Dict[str, Any] = {},
        policy_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        agent_trainer: Optional[AgentTrainer] = None,
        device: str = "auto",
        num_train_steps: int = 100000,
        log_freq: int = 1000,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
    ):
        """Prepares the SCOD trainer for training.

        Args:
            path: Output path.
            scod: SCOD model to train.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            policy_checkpoint: List of policy checkpoints. Either this or
                agent_trainer must be specified.
            agent_trainer: List of agent trainers. Either this or
                policy_checkpoint must be specified.
            device: Torch device.
            num_train_steps: Number of steps to train.
            log_freq: Logging frequency.
            checkpoint: Optional path to trainer checkpoint.
        """
        self._path = pathlib.Path(path) / "scod"
        self._scod = scod
        
        if agent_trainer is None:
            if policy_checkpoint is None:
                raise ValueError(
                    "One of agent_trainer or policy_checkpoint must be specified"
                )
            policy_checkpoint = pathlib.Path(policy_checkpoint)
            if policy_checkpoint.is_file():
                trainer_checkpoint = (
                    policy_checkpoint.parent
                    / policy_checkpoint.name.replace("model", "trainer")
                )
            else:
                trainer_checkpoint = policy_checkpoint / "final_trainer.pt"
            agent_trainer = load_trainer(checkpoint=trainer_checkpoint)
        self._agent_trainer = agent_trainer

        self._dataset = configs.get_class(dataset_class, datasets)(
            observation_space=self.agent.observation_space,
            action_space=self.agent.action_space,
            path=self.path / "train_data", 
            capacity=num_train_steps, 
            **dataset_kwargs
        )

        self.num_train_steps = num_train_steps
        self.log_freq = log_freq

        self._log = logging.Logger(self.path)

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
    def path(self) -> pathlib.Path:
        """Training output path."""
        return self._path

    @property
    def model(self) -> scod.SCOD:
        """SCOD model being trained."""
        return self._scod
    
    @property
    def agent(self) -> agents.RLAgent[ObsType]:
        """Agent being evaluated."""
        return self._agent_trainer.agent
    
    @property
    def env(self) -> envs.Env:
        """Agent env."""
        return self._agent_trainer.env

    @property
    def dataset(self) -> torch.utils.data.IterableDataset:
        """Train dataset."""
        return self._dataset
    
    @property
    def log(self) -> logging.Logger:
        """Tensorboard logger."""
        return self._log

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    def increment_step(self):
        """Increments the training step."""
        self._step += 1

    @property
    def epoch(self) -> int:
        """Current training epoch."""
        return self._epoch
    
    def increment_epoch(self) -> None:
        """Increment the training epoch."""
        self._epoch += 1

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def state_dict(self) -> Dict[str, Any]:
        """Gets the trainer state dicts."""
        state_dict: Dict[str, Any] = {
            "step": self.step,
            "dataset_size": len(self.dataset),
        }
        state_dict["model"] = self.model.state_dict()
        state_dict["agent"] = self.agent.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Loads the trainer state dict.

        Args:
            state_dict: Torch state dict.
            strict: Ensure state_dict keys match networks exactly.
        """
        self._step = state_dict["step"]
        self.model.load_state_dict(state_dict["model"])
        self.agent.load_state_dict(state_dict["agent"])
        
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
        if dataset_size is None:
            dataset_size = state_dict["dataset_size"]
        self.dataset.load(max_entries=dataset_size)

    def to(self, device: Union[str, torch.device]) -> "SCODTrainer":
        """Transfer networks to a device."""
        self._device = tensors.device(device)
        self.model.to(self.device)
        self.agent.to(self.device)
        return self
    
    def train_mode(self) -> None:
        """Switches to training mode."""
        self.model.train_mode()
        self.agent.train_mode()

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        self.model.eval_mode()
        self.agent.eval_mode()

    def collect_step(self, random: bool = False) -> Dict[str, Any]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics.
        """
        if self.step == 0:
            self.dataset.add(observation=self.env.reset())
            self._episode_length = 0
            self._episode_reward = 0

        if random:
            action = self.agent.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                observation = tensors.from_numpy(
                    self.env.get_observation(), self.device
                )
                if not isinstance(self.agent.encoder.network, (OracleEncoder, NormalizeObservation)):
                    observation = tensors.rgb_to_cnn(observation)
                action = self.agent.actor.predict(
                    self.agent.encoder.encode(observation)
                )
                action = action.cpu().numpy()
            self.train_mode()

        next_observation, reward, done, info = self.env.step(action)
        discount = 1.0 - done

        self.dataset.add(
            action=action,
            reward=reward,
            next_observation=next_observation,
            discount=discount,
            done=done,
        )

        self._episode_length += 1
        self._episode_reward += reward
        if not done:
            return {}

        self.increment_epoch()

        metrics = {
            "reward": self._episode_reward,
            "length": self._episode_length,
            "episode": self.epoch
        }

        # Reset the environment
        self.dataset.add(observation=self.env.reset())
        self._episode_length = 0
        self._episode_reward = 0

        return metrics

    def train(self) -> None:
        """Trains the SCOD model."""
        self.dataset.initialize()
        metrics_list = []
        for self._step in range(self.num_train_steps):

            # Collect transition
            metrics_dict = self.collect_step(random=False)
            metrics_list.append(metrics_dict)
            
            # Log metrics
            if self._step % self.log_freq == 0:
                log_metrics = metrics.collect_metrics(metrics_list)
                self.log.log("collect", log_metrics)
                self.log.flush(step=self.step)
                metrics_list = []
        
        # SCOD pre-processing
        self.model.eval_mode()
        self.model.process_dataset(
            self.dataset, 
            input_keys=["observation", "action"],
        )

        self.save(self.path, "final_trainer")
        self.model.save(self.path / "final_scod.pt")
        self.dataset.save()
