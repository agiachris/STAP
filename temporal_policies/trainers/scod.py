import pathlib
from typing import Any, Dict, Mapping, Optional, Union, Type

import torch
import tqdm

from temporal_policies import agents, datasets, envs, scod, trainers
from temporal_policies.networks.encoders import IMAGE_ENCODERS
from temporal_policies.trainers.utils import load as load_trainer
from temporal_policies.utils import configs, tensors, logging, metrics


class SCODReplayBuffer(torch.utils.data.IterableDataset):
    """Wrapper around the ReplayBuffer to encode observations for SCOD."""

    def __init__(
        self,
        agent: agents.RLAgent,
        replay_buffer: datasets.ReplayBuffer,
        batch_size: int = 128,
        sample_strategy: Union[
            str, datasets.ReplayBuffer.SampleStrategy
        ] = "sequential",
        **kwargs,
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.replay_buffer._batch_size = batch_size
        self.replay_buffer._sample_strategy = (
            datasets.ReplayBuffer.SampleStrategy[sample_strategy.upper()]
            if isinstance(sample_strategy, str)
            else sample_strategy
        )

    def __iter__(self):
        self.replay_buffer.initialize()

        for batch in tqdm.tqdm(
            self.replay_buffer,
            total=len(self.replay_buffer) // self.replay_buffer.batch_size,
        ):
            t_observation = tensors.from_numpy(batch["observation"], self.agent.device)
            if isinstance(self.agent.encoder.network, IMAGE_ENCODERS):
                t_observation = tensors.rgb_to_cnn(t_observation)
            t_state = self.agent.encoder.encode(t_observation, batch["policy_args"])

            yield {
                "state": t_state.cpu().numpy(),
                "action": batch["action"],
            }


class SCODTrainer:
    """SCOD trainer."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        scod: scod.SCOD,
        dataset_class: Union[str, Type[datasets.ReplayBuffer]] = datasets.ReplayBuffer,
        dataset_kwargs: Dict[str, Any] = {},
        agent: Optional[agents.RLAgent] = None,
        policy_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        env_kwargs: Dict[str, Any] = {},
        collect_dataset: bool = False,
        device: str = "auto",
        num_train_steps: int = 10000,
        log_freq: int = 100,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
    ):
        """Prepares the SCOD trainer for training.

        Args:
            path: Output path.
            scod: SCOD model to train.
            dataset_class: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            agent: Agent for loading agent trainer.
            policy_checkpoint: Policy checkpoint.
            env_kwargs: Optional kwargs passed to EnvFactory.
            collect_dataset: Whether to collect a new dataset or use the agent's
                saved replay buffer.
            device: Torch device.
            num_train_steps: Number of steps to train.
            log_freq: Logging frequency.
            checkpoint: Optional path to trainer checkpoint.
        """
        if policy_checkpoint is None:
            raise ValueError("Policy_checkpoint must be specified")
        policy_checkpoint = pathlib.Path(policy_checkpoint)
        if policy_checkpoint.is_file():
            trainer_checkpoint = (
                policy_checkpoint.parent
                / policy_checkpoint.name.replace("model", "trainer")
            )
        else:
            trainer_checkpoint = policy_checkpoint / "final_trainer.pt"
        agent_trainer = load_trainer(
            checkpoint=trainer_checkpoint, agent=agent, env_kwargs=env_kwargs
        )
        assert isinstance(agent_trainer, trainers.AgentTrainer)

        self._path = pathlib.Path(path)
        self._scod = scod
        self._agent_trainer = agent_trainer

        self._collect_dataset = collect_dataset
        if self.collect_dataset:
            self._dataset = configs.get_class(dataset_class, datasets)(
                observation_space=self.agent.observation_space,
                action_space=self.agent.action_space,
                path=self.path / "train_data",
                capacity=num_train_steps,
                **dataset_kwargs,
            )
        else:
            self._dataset = agent_trainer.dataset
        self._scod_dataset = SCODReplayBuffer(
            agent_trainer.model, self.dataset, **dataset_kwargs
        )

        self.num_train_steps = num_train_steps
        self.log_freq = log_freq

        self._log = logging.Logger(self.path)

        self._step = 0
        self._epoch = 0

        self.to(device)
        if checkpoint is not None:
            self.load(checkpoint, strict=True)

        self._episode_length = 0
        self._episode_reward = 0.0

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
    def agent(self) -> agents.RLAgent:
        """Agent being evaluated."""
        return self._agent_trainer.agent

    @property
    def env(self) -> envs.Env:
        """Agent env."""
        return self._agent_trainer.env

    @property
    def dataset(self) -> datasets.ReplayBuffer:
        """Train dataset."""
        return self._dataset

    @property
    def collect_dataset(self) -> bool:
        return self._collect_dataset

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

    def collect_step(self, random: bool = False) -> Mapping[str, Any]:
        """Collects data for the replay buffer.

        Args:
            random: Use random actions.

        Returns:
            Collect metrics.
        """
        if self.step == 0:
            observation, _ = self.env.reset()
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
                if isinstance(self.agent.encoder.network, IMAGE_ENCODERS):
                    t_observation = tensors.rgb_to_cnn(t_observation)
                policy_args = self.env.get_primitive().get_policy_args()
                t_action = self.agent.actor.predict(
                    self.agent.encoder.encode(t_observation, policy_args)
                )
                action = t_action.cpu().numpy()
            self.train_mode()

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        discount = 1.0 - float(done)
        try:
            policy_args = info["policy_args"]
        except KeyError:
            policy_args = None

        self.dataset.add(
            action=action,
            reward=reward,
            next_observation=next_observation,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            policy_args=policy_args,
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
        observation, _ = self.env.reset()
        self.dataset.add(observation=observation)
        self._episode_length = 0
        self._episode_reward = 0

        return metrics

    def train(self) -> None:
        """Trains the SCOD model."""
        self.dataset.initialize()
        metrics_list = []

        if self.collect_dataset:
            pbar = tqdm.tqdm(
                range(self.num_train_steps),
                desc="Collect transitions",
                dynamic_ncols=True,
            )
            for _ in pbar:
                # Collect transition
                metrics_dict = self.collect_step(random=False)
                metrics_list.append(metrics_dict)
                pbar.set_postfix({"Reward": metrics_dict["reward"]})
                self.increment_step()

                # Log metrics
                if self.step % self.log_freq == 0:
                    log_metrics = metrics.collect_metrics(metrics_list)
                    self.log.log("collect", log_metrics)
                    self.log.flush(step=self.step)
                    metrics_list = []

        # SCOD pre-processing
        self.model.eval_mode()
        self.model.process_dataset(self._scod_dataset, input_keys=["state", "action"])

        self.save(self.path, "final_trainer")
        self.model.save(self.path / "final_scod.pt")
        self.dataset.save()
