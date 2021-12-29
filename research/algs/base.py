import os
import time
import torch
import numpy as np
import random
import copy

from abc import ABC, abstractmethod
from collections import defaultdict

import research
from research.processors.base import IdentityProcessor
from research.utils.logger import Logger
from research.utils import utils
from research.utils.evaluate import eval_policy


def log_from_dict(logger, loss_lists, prefix):
    keys_to_remove = []
    for loss_name, loss_value in loss_lists.items():
        if isinstance(loss_value, list) and len(loss_value) > 0:
            logger.record(prefix + "/" + loss_name, np.mean(loss_value))
            keys_to_remove.append(loss_name)
        else:
            logger.record(prefix + "/" + loss_name, loss_value)
            keys_to_remove.append(loss_name)
    for key in keys_to_remove:
        del loss_lists[key]

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

MAX_VALID_METRICS = {"reward",}

class Algorithm(ABC):

    def __init__(self, env, network_class, dataset_class, 
                       network_kwargs={}, dataset_kwargs={},
                       device="auto",
                       optim_class=torch.optim.Adam,
                       optim_kwargs={
                           "lr": 0.0001
                       },
                       processor_class=None,
                       processor_kwargs={},
                       checkpoint=None,
                       validation_dataset_kwargs=None,
                       collate_fn=None,
                       batch_size=64,
                       eval_env=None):

        # Save relevant values
        self.env = env
        self.eval_env = eval_env

        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.validation_dataset_kwargs = validation_dataset_kwargs
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        # setup devices
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Setup the data preprocessor first. Thus, if we need to reference it in 
        # network setup we can.
        self.setup_processor(processor_class, processor_kwargs)

        # Create the network
        self.setup_network(network_class, network_kwargs)

        # Create the optimizers
        self.optim = {}
        self.setup_optimizers(optim_class, optim_kwargs)

        # Load a check point if we have one
        if checkpoint:
            self.load(checkpoint, strict=True)

    def setup_processor(self, processor_class, processor_kwargs):
        if processor_class is None:
            self.processor = IdentityProcessor(self.env.observation_space, self.env.action_space)
        else:
            self.processor = processor_class(self.env.observation_space, self.env.action_space, **processor_kwargs)

    def setup_network(self, network_class, network_kwargs):
        self.network = network_class(self.env.observation_space, self.env.action_space, **network_kwargs).to(self.device)

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Default optimizer initialization
        self.optim['network'] = optim_class(self.network.parameters(), **optim_kwargs)

    def setup_datasets(self):
        '''
        Setup the datasets. Note that this is called only during the learn method and thus doesn't take any arguments.
        Everything must be saved apriori. This is done to ensure that we don't need to load all of the data to load the model.
        '''
        self.dataset = self.dataset_class(self.env.observation_space, self.env.action_space, **self.dataset_kwargs)
        if not self.validation_dataset_kwargs is None:
            validation_dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
            validation_dataset_kwargs.update(self.validation_dataset_kwargs)
            self.validation_dataset = self.dataset_class(self.env.observation_space, self.env.action_space, **validation_dataset_kwargs)
        else:
            self.validation_dataset = None

    def save(self, path, extension):
        '''
        Saves a checkpoint of the model and the optimizers
        '''
        optim = {k: v.state_dict() for k, v in self.optim.items()}
        save_dict = {"network" : self.network.state_dict(), "optim": optim}
        torch.save(save_dict, os.path.join(path, extension + ".pt"))

    def load(self, checkpoint, initial_lr=None, strict=True):
        '''
        Loads the model and its associated checkpoints.
        '''
        print("[research] loading checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'], strict=strict)
        
        if strict:
            # Only load the optimizer state dict if we are being strict.
            for k, v in self.optim.items():
                self.optim[k].load_state_dict(checkpoint['optim'][k])
        
        # make sure that we reset the learning rate in case we decide to not use scheduling for finetuning.
        if not initial_lr is None:
            for param_group in self.optim.param_groups:
                param_group['lr'] = initial_lr

    @property
    def steps(self):
        return self._steps

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
            batch = utils.to_device(batch)
        return batch

    def train(self, path, total_steps, schedule=None, schedule_kwargs={}, log_freq=100, eval_freq=1000, max_eval_steps=-1, workers=4, loss_metric="loss", eval_ep=-1):
        logger = Logger(path=path)
        print("[research] Model Directory:", path)
        print("[research] Training a model with", sum(p.numel() for p in self.network.parameters() if p.requires_grad), "trainable parameters.")
        
        # Construct the dataloaders.
        self.setup_datasets()
        shuffle = not issubclass(self.dataset_class, torch.utils.data.IterableDataset)
        pin_memory = self.device.type == "cuda"
        worker_init_fn = _worker_init_fn if workers > 0 else None
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                                 shuffle=shuffle, 
                                                 num_workers=workers, worker_init_fn=worker_init_fn,
                                                 pin_memory=pin_memory, 
                                                 collate_fn=self.collate_fn)
        if self.validation_dataset is not None:
            validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, 
                                                            shuffle=shuffle, 
                                                            num_workers=workers, 
                                                            pin_memory=pin_memory,
                                                            collate_fn=self.collate_fn)
        else:
            validation_dataloader = None

        # Create schedulers for the optimizers
        schedulers = {}
        if schedule is not None:
            for name, opt in self.optim.items():
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.schedule_fn(total_steps, **schedule_kwargs))

        # Setup model metrics.
        self._steps = 0
        epochs = 0
        loss_lists = defaultdict(list)
        best_validation_metric = -1*float('inf') if loss_metric in MAX_VALID_METRICS else float('inf')
        start_time = time.time()

        self.network.train()
        
        while self._steps < total_steps:

            for batch in dataloader:
                batch = self._format_batch(batch)

                assert self.network.training, "Network was not in training mode and trainstep was called."
                losses = self._train_step(batch)
                for loss_name, loss_value in losses.items():
                    loss_lists[loss_name].append(loss_value)
                self._steps += 1

                # Update the schedulers
                for scheduler in schedulers.values():
                    scheduler.step()

                # Run the logger
                if self._steps % log_freq == 0:
                    current_time = time.time()
                    log_from_dict(logger, loss_lists, "train")
                    logger.record("time/epochs", epochs)
                    logger.record("time/steps_per_second", log_freq / (current_time - start_time))
                    for name, scheduler in schedulers.items():
                        logger.record("lr/" + name, scheduler.get_last_lr()[0])
                    start_time = current_time
                    logger.dump(step=self._steps)

                if self._steps % eval_freq == 0:
                    self.network.eval()
                    current_validation_metric = None
                    if not validation_dataloader is None:
                        eval_steps = 0
                        validation_loss_lists = defaultdict(list)
                        with torch.no_grad():
                            for batch in validation_dataloader:
                                batch = self._format_batch(batch)
                                losses = self._validation_step(batch)
                                for loss_name, loss_value in losses.items():
                                    validation_loss_lists[loss_name].append(loss_value)
                                eval_steps += 1
                                if eval_steps == max_eval_steps:
                                    break

                        if loss_metric in validation_loss_lists:
                            current_validation_metric = np.mean(validation_loss_lists[loss_metric])
                        log_from_dict(logger, validation_loss_lists, "valid")

                    # Now run any extra validation steps, independent of the validation dataset.
                    validation_extras = self._validation_extras(path, self._steps, validation_dataloader)
                    if loss_metric in validation_extras:
                        current_validation_metric = validation_extras[loss_metric]
                    log_from_dict(logger, validation_extras, "valid")

                    # TODO: evaluation episodes.
                    if self.eval_env is not None and eval_ep > 0:
                        eval_metrics = eval_policy(self.eval_env, self, eval_ep)
                        if loss_metric in eval_metrics:
                            current_validation_metric = eval_metrics[loss_metric]
                        log_from_dict(logger, eval_metrics, "eval")

                    if loss_metric in MAX_VALID_METRICS and current_validation_metric > best_validation_metric:
                        self.save(path, "best_model")
                        best_validation_metric = current_validation_metric
                    elif current_validation_metric < best_validation_metric:
                        self.save(path, "best_model")
                        best_validation_metric = current_validation_metric

                    # Eval Logger Dump to CSV
                    logger.dump(step=self._steps, dump_csv=True) # Dump the eval metrics to CSV.
                    self.save(path, "final_model") # Also save the final model every eval period.
                    self.network.train()

                if self._steps >= total_steps:
                    break
                
            epochs += 1

    @abstractmethod
    def _train_step(self, batch):
        '''
        Train the model. Should return a dict of loggable values
        '''
        pass

    @abstractmethod
    def _validation_step(self, batch):
        '''
        perform a validation step. Should return a dict of loggable values.
        '''
        pass

    def _validation_extras(self, path, step, dataloader):
        '''
        perform any extra validation operations
        '''
        return {}

    def _predict(self, batch):
        '''Internal prediction function, can be overridden'''
        if hasattr(self.network, "predict"):
            pred = self.network.predict(batch)
        else:
            pred = self.network(batch)
        return batch

    def predict(self, batch, is_batched=False):
        is_np = not utils.contains_tensors(batch)
        if not is_batched:
            # Unsqeeuze everything
            batch = utils.unsqueeze(batch, 0)
        batch = self._format_batch(batch)
        pred = self._predict(batch)
        if not is_batched:
            pred = utils.get_from_batch(pred, 0)
        if is_np:
            pred = utils.to_np(pred)
        return pred
