import abc
from collections import defaultdict
import os
import random
import pathlib
import time
from typing import Any, Dict, Optional, Type, Union

import numpy as np  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies import datasets, envs, networks, processors
from temporal_policies.agents import base as agents
from temporal_policies.utils.logger import Logger
from temporal_policies.utils import configs, utils
from temporal_policies.utils.evaluate import eval_policy


def log_from_dict(logger, loss_lists, prefix, log_stddev: bool = False):
    keys_to_remove = []
    for loss_name, loss_value in loss_lists.items():
        if isinstance(loss_value, list) and len(loss_value) > 0:
            logger.record(prefix + "/" + loss_name, np.mean(loss_value))
            if log_stddev:
                logger.record(f"{prefix}/{loss_name}/stddev", np.std(loss_value))
            keys_to_remove.append(loss_name)
        else:
            logger.record(prefix + "/" + loss_name, loss_value)
            if log_stddev:
                logger.record(f"{prefix}/{loss_name}/stddev", 0)
            keys_to_remove.append(loss_name)
    for key in keys_to_remove:
        del loss_lists[key]


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


MAX_VALID_METRICS = {"reward", "accuracy"}


class RLAgent(agents.Agent):
    def __init__(
        self,
        env: envs.Env,
        network_class: Union[str, Type[torch.nn.Module]],
        network_kwargs: Dict[str, Any],
        dataset_class: Union[str, Type[torch.utils.data.IterableDataset]],
        dataset_kwargs: Dict[str, Any],
        eval_dataset_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 0.0001},
        processor_class: Optional[Union[str, Type[processors.Processor]]] = None,
        processor_kwargs: Dict[str, Any] = {},
        batch_size: int = 64,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
        collate_fn=None,
    ):
        self.dataset_class = configs.get_class(dataset_class, datasets)
        self.dataset_kwargs = dataset_kwargs
        self.eval_dataset_kwargs = (
            eval_dataset_kwargs
            if eval_dataset_kwargs is not None
            else dict(dataset_kwargs)
        )
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        # setup devices
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Setup the data preprocessor first. Thus, if we need to reference it in
        # network setup we can.
        if processor_class is None:
            processor_class = processors.IdentityProcessor
        else:
            processor_class = configs.get_class(processor_class, processors)
        self.processor = processor_class(
            env.observation_space, env.action_space, **processor_kwargs
        )

        # self._state_space = env.state_space
        self._action_space = env.action_space
        self._observation_space = env.observation_space

        network_class = configs.get_class(network_class, networks)
        self.setup_network(network_class, network_kwargs)

        # Create the optimizers
        self.optim: Dict[str, torch.optim.Optimizer] = {}
        optimizer_class = configs.get_class(optimizer_class, torch.optim)
        self.setup_optimizers(optimizer_class, optimizer_kwargs)

        # Load a check point if we have one
        if checkpoint is not None:
            self._path = pathlib.Path(checkpoint).parent
            self.load(checkpoint, strict=True)

        super().__init__(
            state_space=env.observation_space,
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor=self.network.actor,
            critic=self.network.critic,
            encoder=self.network.encoder,
            device=device,
        )

    def setup_network(self, network_class, network_kwargs):
        # Create the network
        self.network = network_class(
            self._env.observation_space, self._env.action_space, **network_kwargs
        )

    def setup_optimizers(self, optimizer_class, optimizer_kwargs):
        self.optim = {
            "network": optimizer_class(self.network.parameters(), **optimizer_kwargs)
        }

    def setup_datasets(self, path: Optional[pathlib.Path] = None):
        """
        Setup the datasets. Note that this is called only during the learn method and thus doesn't take any arguments.
        Everything must be saved apriori. This is done to ensure that we don't need to load all of the data to load the model.
        """
        if path is None:
            path = self._path
        self.dataset = self.dataset_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            path=path / "train_data",
            **self.dataset_kwargs,
        )
        self.eval_dataset = self.dataset_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            path=path / "eval_data",
            **self.eval_dataset_kwargs,
        )
        self.eval_dataset.initialize()
        self.validation_dataset = None
        # if self.validation_dataset_kwargs is not None:
        #     validation_dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
        #     validation_dataset_kwargs.update(self.validation_dataset_kwargs)
        #     self.validation_dataset = self.dataset_class(
        #         observation_space=self.observation_space,
        #         action_space=self.action_space,
        #         path=path / "eval_data",
        #         **validation_dataset_kwargs,
        #     )
        # else:
        #     self.validation_dataset = None

    def save(self, path, extension):
        """
        Saves a checkpoint of the model and the optimizers
        """
        optim = {k: v.state_dict() for k, v in self.optim.items()}
        save_dict = {"network": self.network.state_dict(), "optim": optim}
        torch.save(save_dict, os.path.join(path, extension + ".pt"))

    def load(self, checkpoint, initial_lr=None, strict=True):
        """
        Loads the model and its associated checkpoints.
        """
        print("[research] loading checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"], strict=strict)

        if strict:
            # Only load the optimizer state dict if we are being strict.
            for k, v in self.optim.items():
                self.optim[k].load_state_dict(checkpoint["optim"][k])

        # make sure that we reset the learning rate in case we decide to not use scheduling for finetuning.
        if initial_lr is not None:
            for param_group in self.optim.param_groups:
                param_group["lr"] = initial_lr

    @property
    def steps(self):
        return self._steps

    @property
    def total_steps(self):
        if hasattr(self, "_total_steps"):
            return self._total_steps
        else:
            raise ValueError(
                "alg.train has not been called, no total step count available."
            )

    def _format_batch(self, batch):
        # Convert items to tensor if they are not.
        if not utils.contains_tensors(batch):
            batch = utils.to_tensor(batch)
        if self.processor.supports_gpu:
            # Move to CUDA first.
            batch = utils.to_device(batch, self.device)
            batch = self.processor(batch)
        else:
            batch = self.processor(batch)
            batch = utils.to_device(batch, self.device)
        return batch

    def train(
        self,
        env: envs.Env,
        path,
        total_steps,
        eval_env: Optional[envs.Env] = None,
        schedule=None,
        schedule_kwargs={},
        log_freq=100,
        eval_freq=1000,
        max_eval_steps=-1,
        workers=4,
        loss_metric="loss",
        eval_ep=-1,
        profile_freq=-1,
    ):
        logger = Logger(path=path)
        print("[research] Model Directory:", path)
        print(
            "[research] Training a model with",
            sum(p.numel() for p in self.network.parameters() if p.requires_grad),
            "trainable parameters.",
        )

        # Construct the dataloaders.
        self.setup_datasets(pathlib.Path(path))
        shuffle = not issubclass(self.dataset_class, torch.utils.data.IterableDataset)
        pin_memory = self.device.type == "cuda"
        worker_init_fn = _worker_init_fn if workers > 0 else None
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )
        if self.validation_dataset is not None:
            validation_dataloader = torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=self.collate_fn,
            )
        else:
            validation_dataloader = None

        # Create schedulers for the optimizers
        schedulers = {}
        if schedule is not None:
            for name, opt in self.optim.items():
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                    opt, lr_lambda=self.schedule_fn(total_steps, **schedule_kwargs)
                )

        # Setup model metrics.
        self._steps = 0
        self._total_steps = total_steps
        epochs = 0
        loss_lists = defaultdict(list)
        best_validation_metric = (
            -1 * float("inf") if loss_metric in MAX_VALID_METRICS else float("inf")
        )

        # Setup profiling
        start_time = current_time = time.time()
        profiling_lists = defaultdict(list)

        self.network.train()

        with tqdm.tqdm(total=total_steps) as pbar:
            while self._steps < total_steps:
                for batch in dataloader:
                    # Profiling
                    if profile_freq > 0 and self._steps % profile_freq == 0:
                        stop_time = time.time()
                        profiling_lists["dataset"].append(stop_time - current_time)
                        current_time = stop_time

                    batch = self._format_batch(batch)

                    if profile_freq > 0 and self._steps % profile_freq == 0:
                        stop_time = time.time()
                        profiling_lists["preprocess"].append(stop_time - current_time)
                        current_time = stop_time

                    # Train the network
                    assert (
                        self.network.training
                    ), "Network was not in training mode and trainstep was called."
                    losses = self._train_step(env, batch)
                    for loss_name, loss_value in losses.items():
                        loss_lists[loss_name].append(loss_value)

                    if profile_freq > 0 and self._steps % profile_freq == 0:
                        stop_time = time.time()
                        profiling_lists["train_step"].append(stop_time - current_time)

                    # Increment the number of training steps.
                    self._steps += 1

                    # Update the schedulers
                    for scheduler in schedulers.values():
                        scheduler.step()

                    # Run the logger
                    if self._steps % log_freq == 0:
                        current_time = time.time()
                        log_from_dict(logger, loss_lists, "train")
                        logger.record("time/epochs", epochs)
                        logger.record(
                            "time/steps_per_second",
                            log_freq / (current_time - start_time),
                        )
                        log_from_dict(logger, profiling_lists, "time")
                        for name, scheduler in schedulers.items():
                            logger.record("lr/" + name, scheduler.get_last_lr()[0])
                        start_time = current_time
                        logger.dump(step=self._steps)

                    if self._steps % eval_freq == 0:
                        self.eval_mode()
                        current_validation_metric = None
                        if validation_dataloader is not None:
                            eval_steps = 0
                            validation_loss_lists = defaultdict(list)
                            for batch in validation_dataloader:
                                batch = self._format_batch(batch)
                                losses = self._validation_step(batch)
                                for loss_name, loss_value in losses.items():
                                    validation_loss_lists[loss_name].append(loss_value)
                                eval_steps += 1
                                if eval_steps == max_eval_steps:
                                    break

                            if loss_metric in validation_loss_lists:
                                current_validation_metric = np.mean(
                                    validation_loss_lists[loss_metric]
                                )
                            log_from_dict(logger, validation_loss_lists, "valid")

                        # Now run any extra validation steps, independent of the validation dataset.
                        validation_extras = self._validation_extras(
                            path, self._steps, validation_dataloader
                        )
                        if loss_metric in validation_extras:
                            current_validation_metric = validation_extras[loss_metric]
                        log_from_dict(logger, validation_extras, "valid")

                        # TODO: evaluation episodes.
                        if eval_env is not None and eval_ep > 0:
                            eval_metrics = eval_policy(
                                eval_env, self, eval_ep, self.eval_dataset
                            )
                            if loss_metric in eval_metrics:
                                current_validation_metric = eval_metrics[loss_metric]
                            log_from_dict(logger, eval_metrics, "eval")

                        if current_validation_metric is None:
                            pass
                        elif (
                            loss_metric in MAX_VALID_METRICS
                            and current_validation_metric > best_validation_metric
                        ):
                            self.save(path, "best_model")
                            best_validation_metric = current_validation_metric
                        elif current_validation_metric < best_validation_metric:
                            self.save(path, "best_model")
                            best_validation_metric = current_validation_metric

                        # Eval Logger Dump to CSV
                        logger.dump(
                            step=self._steps, dump_csv=True
                        )  # Dump the eval metrics to CSV.
                        self.save(
                            path, "final_model"
                        )  # Also save the final model every eval period.
                        self.train_mode()

                    # Profiling
                    if profile_freq > 0 and self._steps % profile_freq == 0:
                        current_time = time.time()

                    pbar.update(1)
                    if self._steps >= total_steps:
                        break

                epochs += 1

    @abc.abstractmethod
    def _train_step(self, env: envs.Env, batch):
        """
        Train the model. Should return a dict of loggable values
        """
        pass

    @abc.abstractmethod
    def _validation_step(self, batch):
        """
        perform a validation step. Should return a dict of loggable values.
        """
        pass

    def _validation_extras(self, path, step, dataloader):
        """
        perform any extra validation operations
        """
        return {}

    def train_mode(self):
        self.network.train()
        self.processor.train()

    def eval_mode(self):
        self.network.eval()
        self.processor.eval()

    def _predict(self, batch, **kwargs):
        """
        Internal prediction function, can be overridden
        By default, we call torch.no_grad(). If this behavior isn't desired,
        override the _predict funciton in your algorithm.
        """
        with torch.no_grad():
            if hasattr(self.network, "predict"):
                pred = self.network.predict(batch, **kwargs)
            else:
                if len(kwargs) > 0:
                    raise ValueError(
                        "Default predict method does not accept key word args, but they were provided."
                    )
                pred = self.network(batch)
        return pred

    def predict(self, batch, is_batched=False, **kwargs):
        is_np = not utils.contains_tensors(batch)
        if not is_batched:
            # Unsqeeuze everything
            batch = utils.unsqueeze(batch, 0)
        batch = self._format_batch(batch)
        pred = self._predict(batch, **kwargs)
        if not is_batched:
            pred = utils.get_from_batch(pred, 0)
        if is_np:
            pred = utils.to_np(pred)
        return pred

    def to(self, device: Union[str, torch.device]) -> agents.Agent:
        """Transfers networks to device."""
        super().to(device)
        self.network.to(self.device)
        return self

    @property
    def critic(self) -> torch.nn.Module:
        return self.network.critic

    @property
    def actor(self) -> torch.nn.Module:
        return self.network.actor
