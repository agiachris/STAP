import abc
import pathlib
from typing import Any, Dict, Generic, Mapping, Type, TypedDict, TypeVar, Union

import numpy as np  # type: ignore
import torch  # type: ignore

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

    @abc.abstractmethod
    def save(self, path: Union[str, pathlib.Path], name: str) -> None:
        pass

    @abc.abstractmethod
    def to(self, device: Union[str, torch.device]) -> "Model":
        pass

    @abc.abstractmethod
    def train_mode(self) -> None:
        pass

    @abc.abstractmethod
    def eval_mode(self) -> None:
        pass


ModelType = TypeVar("ModelType", bound=Model)


class Batch(TypedDict, Generic[ArrayType, ObsType]):
    observation: ObsType
    action: ArrayType
    reward: ArrayType
    next_observation: ObsType
    discount: ArrayType


class WrappedBatch(Batch):
    idx_replay_buffer: np.ndarray


class DynamicsBatch(TypedDict, Generic[ArrayType, ObsType]):
    observation: ObsType
    idx_policy: ArrayType
    action: ArrayType
    next_observation: ObsType
