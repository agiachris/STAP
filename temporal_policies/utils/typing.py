import abc
import pathlib
from typing import Any, Dict, Generic, Mapping, Type, TypedDict, TypeVar, Union

import numpy as np
import torch

Scalar = Union[np.generic, float, int, bool]
scalars = (np.generic, float, int, bool)


ArrayType = TypeVar("ArrayType", np.ndarray, torch.Tensor)
StateType = TypeVar("StateType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
BatchType = TypeVar("BatchType", bound=Mapping)


class Model(abc.ABC, Generic[BatchType]):
    @abc.abstractmethod
    def create_optimizers(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.optim.Optimizer]:
        pass

    @abc.abstractmethod
    def train_step(
        self,
        step: int,
        batch: BatchType,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        pass

    def save(self, path: Union[str, pathlib.Path], name: str):
        """Saves a checkpoint of the model and the optimizers.

        Args:
            path: Directory of checkpoint.
            name: Name of checkpoint (saved as `path/name.pt`).
        """
        torch.save(self.state_dict(), pathlib.Path(path) / f"{name}.pt")

    def load(self, checkpoint: Union[str, pathlib.Path], strict: bool = True) -> None:
        """Loads the model from the given checkpoint.

        Args:
            checkpoint: Checkpoint path.
            strict: Make sure the state dict keys match.
        """
        device = self.device if hasattr(self, "device") else None  # type: ignore
        state_dict = torch.load(checkpoint, map_location=device)
        self.load_state_dict(state_dict)

    @abc.abstractmethod
    def train_mode(self) -> None:
        pass

    @abc.abstractmethod
    def eval_mode(self) -> None:
        pass

    @abc.abstractmethod
    def to(self, device):
        pass


ModelType = TypeVar("ModelType", bound=Model)


Batch = Dict[str, ArrayType]
# class Batch(TypedDict, Generic[ArrayType, ObsType]):
#     observation: ObsType
#     action: ArrayType
#     reward: ArrayType
#     next_observation: ObsType
#     discount: ArrayType


class WrappedBatch(Batch):
    idx_replay_buffer: np.ndarray


class DynamicsBatch(TypedDict, Generic[ArrayType, ObsType]):
    observation: ObsType
    idx_policy: ArrayType
    action: ArrayType
    next_observation: ObsType


class StateBatch(TypedDict, Generic[ArrayType]):
    state: ArrayType
    observation: ArrayType
    image: ArrayType


class AutoencoderBatch(TypedDict, Generic[ObsType]):
    observation: ObsType


class StateEncoderBatch(TypedDict, Generic[ArrayType, ObsType]):
    observation: ObsType
    state: ArrayType
