#!/usr/bin/env python3

import datetime
import functools
import pathlib
from typing import Generator, Optional, Sequence, TypedDict, Union

import gym
import numpy as np
import torch
import tqdm

from temporal_policies.datasets.replay_buffer import (
    ReplayBuffer,
    _wrap_get,
    _wrap_insert,
)
from temporal_policies.utils import nest, spaces
from temporal_policies.utils.typing import StateBatch


class StorageBatch(TypedDict):
    state: np.ndarray
    observation: np.ndarray
    image: np.ndarray


class StateBuffer(ReplayBuffer):
    """Replay buffer class."""

    def __init__(
        self,
        state_space: gym.spaces.Box,
        observation_space: gym.spaces.Box,
        image_space: gym.spaces.Box,
        path: Optional[Union[str, pathlib.Path]] = None,
        capacity: int = 100000,
        batch_size: Optional[int] = None,
        sample_strategy: Union[str, ReplayBuffer.SampleStrategy] = "uniform",
        save_frequency: Optional[int] = None,
    ):
        """Stores the configuration parameters for the replay buffer.

        The actual buffers will be constructed upon calling
        `ReplayBuffer.__iter__() or `ReplayBuffer.initialize()`.

        Args:
            state_space: Full state space.
            observation_space: Low-dimensional observation space.
            image_space: Image observation space.
            path: Optional location of replay buffer on disk.
            capacity: Replay buffer capacity.
            batch_size: Sample batch size.
            sample_strategy: Sample strategy.
            save_frequency: Frequency of optional automatic saving to disk.
        """
        self._state_space = state_space
        self._observation_space = observation_space
        self._image_space = image_space
        self._capacity = capacity

        self._batch_size = batch_size
        self._sample_strategy = (
            ReplayBuffer.SampleStrategy[sample_strategy.upper()]
            if isinstance(sample_strategy, str)
            else sample_strategy
        )

        self._path = None if path is None else pathlib.Path(path)
        if save_frequency is not None and save_frequency <= 0:
            save_frequency = None
        self._save_frequency = save_frequency

    @property
    def state_space(self) -> gym.spaces.Box:
        """Batch state space."""
        return self._state_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Batch observation space."""
        return self._observation_space

    @property
    def image_space(self) -> gym.spaces.Box:
        """Batch image space."""
        return self._image_space

    @property
    def capacity(self) -> int:
        """Replay buffer capacity."""
        return self._capacity

    def __len__(self) -> int:
        """Number of entries added to the replay buffer."""
        return self._worker_idx

    @property
    def batch_size(self) -> Optional[int]:
        """Sample batch size."""
        return self._batch_size

    @property
    def sample_strategy(self) -> ReplayBuffer.SampleStrategy:
        """Sample strategy."""
        return self._sample_strategy

    @property
    def path(self) -> Optional[pathlib.Path]:
        """Location of replay buffer on disk."""
        return self._path

    @path.setter
    def path(self, path: Union[str, pathlib.Path]) -> None:
        """Sets the location fo replay buffer on disk."""
        self._path = pathlib.Path(path)

    @property
    def save_frequency(self) -> Optional[int]:
        """Frequency of automatic saving to disk."""
        return self._save_frequency

    @property
    def num_workers(self) -> int:
        """Number of parallel data workers."""
        worker_info = torch.utils.data.get_worker_info()
        return 1 if worker_info is None else worker_info.num_workers

    @property
    def worker_id(self) -> int:
        """Current worker id."""
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id

    @property
    def worker_capacity(self) -> int:
        """Current worker capacity."""
        try:
            return self._worker_capacity
        except AttributeError:
            raise RuntimeError("Need to run ReplayBuffer.initialize() first.")

    @property
    def worker_buffers(self):
        """Current worker buffers."""
        try:
            return self._worker_buffers
        except AttributeError:
            raise RuntimeError("Need to run ReplayBuffer.initialize() first.")

    def initialize(self) -> None:
        """Initializes the worker buffers."""

        # Set up only once.
        if hasattr(self, "_worker_buffers"):
            return

        # TODO: Need to think about how to load data among multiple workers when
        # multiple policies are being trained.
        if self.num_workers != 1:
            raise NotImplementedError("Multiple workers not supported yet.")

        self._worker_capacity = self.capacity // self.num_workers
        self._worker_buffers = self.create_default_batch(self.worker_capacity)  # type: ignore
        self._worker_size = 0
        self._worker_idx = 0
        self._worker_idx_checkpoint = 0

    def create_default_batch(self, size: int) -> StorageBatch:  # type: ignore
        """Creates a batch of the specified size with default values.

        Args:
            size: Batch size.

        Returns:
            Batch dict with observation, action, reward, discount, done fields.
        """
        return {
            "state": spaces.null(self.state_space, size),
            "observation": spaces.null(self.observation_space, size),
            "image": spaces.null(self.image_space, size),
        }

    def add(  # type: ignore
        self,
        state: Optional[np.ndarray] = None,
        observation: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        batch: Optional[StorageBatch] = None,
        max_entries: Optional[int] = None,
    ) -> int:
        """Adds an experience tuple to the replay buffer.

        The experience can either be a single initial `observation`, a 5-tuple
        (`action`, `reward`, `next_observation`, `discount`, `done`), or a
        `batch` dict from buffer storage.

        The inputs can be single entries or batches.

        Args:
            state: Full environment state.
            observation: Low-dimensional observation.
            image: Image observation.
            batch: Batch dict. Useful for loading from disk.
            max_entries: Limit the number of entries to add.

        Returns:
            Number of entries added.
        """
        if sum(arg is not None for arg in (observation, batch)) != 1:
            raise ValueError("Only one of observation or batch can be specified.")
        if not ((state is None) == (observation is None) == (image is None)):
            raise ValueError("(state, observation, image) need to be set together.")

        # Prepare batch.
        if batch is None:
            assert state is not None
            assert observation is not None
            assert image is not None
            batch = {
                "state": state,
                "observation": observation,
                "image": image,
            }

        # Insert batch and advance indices.
        idx_start = self._worker_idx
        num_added_structure = nest.map_structure(
            functools.partial(_wrap_insert, idx=idx_start, max_entries=max_entries),
            self.worker_buffers,
            batch,
            atom_type=np.ndarray,
        )
        num_added = next(nest.structure_iterator(num_added_structure, atom_type=int))
        idx_stop = idx_start + num_added

        self._worker_idx = idx_stop
        self._worker_size = min(self.worker_capacity, idx_stop)

        # Save checkpoint.
        len_checkpoint = self._worker_idx - self._worker_idx_checkpoint
        if self.save_frequency is not None and len_checkpoint >= self.save_frequency:
            self.save()

        return num_added

    def sample(  # type: ignore
        self,
        sample_strategy: Optional[ReplayBuffer.SampleStrategy] = None,
        batch_size: Optional[int] = None,
    ) -> Optional[StateBatch]:
        """Samples a batch from the replay buffer.

        Args:
            sample_strategy: Optional sample strategy.
            batch_size: Optional batch size. Otherwise uses default batch size.

        Returns:
            Sample batch.
        """
        if sample_strategy is None:
            sample_strategy = self.sample_strategy
        if batch_size is None:
            batch_size = self.batch_size

        # Get sample indices.
        len_buffer = self._worker_size
        if sample_strategy == ReplayBuffer.SampleStrategy.SEQUENTIAL:
            idx_start = self._idx_deterministic
            num_entries = min(
                1 if batch_size is None else batch_size, len_buffer - idx_start
            )
            if num_entries <= 0:
                return None

            idx_sample = np.arange(idx_start, idx_start + num_entries)
            self._idx_deterministic += num_entries
        else:
            if len_buffer == 0:
                return None
            idx_sample = np.random.randint(len_buffer, size=batch_size)

        # Assemble sample dict.
        state = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample),
            self.worker_buffers["state"],
            atom_type=np.ndarray,
        )
        observation = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample),
            self.worker_buffers["observation"],
            atom_type=np.ndarray,
        )
        image = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample),
            self.worker_buffers["image"],
            atom_type=np.ndarray,
        )

        return StateBatch(
            state=state,
            observation=observation,
            image=image,
        )

    def load(
        self, path: Optional[pathlib.Path] = None, max_entries: Optional[int] = None
    ) -> int:
        """Loads replay buffer checkpoints from disk.

        Args:
            path: Location of checkpoints.
            max_entries: Maximum number of entries to load.

        Returns:
            Number of entries loaded.
        """
        if path is None:
            path = self.path
        if path is None:
            return 0

        # TODO: Can't be initialized unless this is already a worker thread.
        self.initialize()

        num_loaded = 0
        checkpoint_paths = sorted(
            path.iterdir(), key=lambda f: tuple(map(int, f.stem.split("_")[:-1]))
        )
        pbar = tqdm.tqdm(checkpoint_paths, desc=f"Load {path}", dynamic_ncols=True)
        for checkpoint_path in pbar:
            with open(checkpoint_path, "rb") as f:
                checkpoint: StorageBatch = dict(np.load(f))  # type: ignore
            num_added = self.add(batch=checkpoint, max_entries=max_entries)
            num_loaded += num_added

            if max_entries is not None:
                max_entries -= num_added
                if max_entries <= 0:
                    break

        return num_loaded

    def save(self, path: Optional[pathlib.Path] = None) -> int:
        """Saves a replay buffer checkpoint to disk.

        The checkpoint filename is saved as
        "{timestamp}_{worker_id}_{checkpoint_size}.npz".

        Args:
            path: Location of checkpoints.

        Returns:
            Number of entries saved.
        """
        if path is None:
            path = self.path
        if path is None:
            return 0

        idx_start = self._worker_idx_checkpoint
        idx_stop = self._worker_idx
        if idx_stop < idx_start:
            idx_stop += self.worker_capacity
        checkpoint = self[idx_start:idx_stop]
        len_checkpoint = idx_stop - idx_start

        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        checkpoint_name = f"{timestamp}_{self.worker_id}_{len_checkpoint}"
        with open(path / f"{checkpoint_name}.npz", "wb") as f:
            np.savez_compressed(f, **checkpoint)

        self._worker_idx_checkpoint = idx_stop

        return len_checkpoint

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]):
        """Gets the given entries from the buffers.

        Args:
            idx: Numpy-style indices.

        Returns:
            Buffer slices. May be mutable views of the original buffers or
            temporary copies.
        """
        is_invalid = False
        if isinstance(idx, int):
            is_invalid = idx >= self._worker_idx
        elif isinstance(idx, slice):
            is_invalid = idx.start is not None and idx.start >= self._worker_idx
            if idx.stop is not None and idx.stop > self._worker_idx:
                idx = slice(idx.start, self._worker_idx, idx.step)
        else:
            is_invalid = any(i >= self._worker_idx for i in idx)
        if is_invalid:
            raise ValueError(f"Cannot index beyond {self._worker_idx}: idx={idx}.")

        return nest.map_structure(
            functools.partial(_wrap_get, idx=idx),
            self.worker_buffers,
            atom_type=np.ndarray,
        )

    def __iter__(self) -> Generator[StateBatch, None, None]:  # type: ignore
        """Iterates over the replay buffer."""
        self.initialize()

        if self.sample_strategy == ReplayBuffer.SampleStrategy.SEQUENTIAL:
            self._idx_deterministic = 0

        while True:
            sample = self.sample()
            if sample is None:
                return
            yield sample


# if __name__ == "__main__":
#     # Simple tests.
#     observation_space = gym.spaces.Box(low=np.full(2, 0), high=np.full(2, 1))
#     action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
#     replay_buffer = ReplayBuffer[np.ndarray](
#         observation_space, action_space, capacity=5, batch_size=4
#     )
#
#     replay_buffer.initialize()
#     print(replay_buffer.worker_buffers)
#     print("SAMPLE", replay_buffer.sample())
#
#     for i in range(3):
#         v = 0.2 * i
#         replay_buffer.add(observation=np.full(2, v))
#         print("")
#         print(replay_buffer.worker_buffers)
#         print("SAMPLE", replay_buffer.sample())
#
#         replay_buffer.add(
#             next_observation=np.full(2, v + 0.1),
#             action=np.full(1, v + 0.1),
#             reward=v,
#             discount=0.99,
#             done=True,
#         )
#         print("")
#         print(replay_buffer.worker_buffers)
#         print("SAMPLE", replay_buffer.sample())
#
#     for i, batch in enumerate(replay_buffer):
#         if i > 2:
#             break
#         print("BATCH", batch)
#         print("")
#     # print("SAMPLE", replay_buffer.sample())
