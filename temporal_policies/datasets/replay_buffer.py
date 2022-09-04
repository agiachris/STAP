#!/usr/bin/env python3


import datetime
import enum
import functools
import pathlib
from typing import Any, Generator, Optional, Sequence, Union

try:
    from typing import TypedDict
except ModuleNotFoundError:
    from typing_extensions import TypedDict

import gym
import numpy as np
import torch
import tqdm

from temporal_policies.utils import nest, spaces
from temporal_policies.utils.typing import Batch


class StorageBatch(TypedDict):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    policy_args: np.ndarray


class ReplayBuffer(torch.utils.data.IterableDataset):
    """Replay buffer class."""

    class SampleStrategy(enum.Enum):
        """Replay buffer sample strategy."""

        UNIFORM = 0  # Uniform random sampling.
        SEQUENTIAL = 1  # Deterministic sequential order.

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        path: Optional[Union[str, pathlib.Path]] = None,
        capacity: int = 100000,
        batch_size: Optional[int] = None,
        sample_strategy: Union[str, SampleStrategy] = "uniform",
        nstep: int = 1,
        save_frequency: Optional[int] = None,
        skip_truncated: bool = False,
        skip_failed: bool = False,
    ):
        """Stores the configuration parameters for the replay buffer.

        The actual buffers will be constructed upon calling
        `ReplayBuffer.__iter__() or `ReplayBuffer.initialize()`.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            path: Optional location of replay buffer on disk.
            capacity: Replay buffer capacity.
            batch_size: Sample batch size.
            sample_strategy: Sample strategy.
            nstep: Number of steps between sample and next observation.
            save_frequency: Frequency of optional automatic saving to disk.
            skip_truncated: Whether to mark truncated episodes as invalid when
                adding to the replay buffer. If true, truncated episodes won't
                be sampled.
            skip_failed: Whether to mark reward < 1 episodes as invalid when
                adding to the replay buffer. If true, failed episodes won't be
                sampled.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._capacity = capacity

        self._batch_size = batch_size
        self._sample_strategy = (
            ReplayBuffer.SampleStrategy[sample_strategy.upper()]
            if isinstance(sample_strategy, str)
            else sample_strategy
        )
        self._nstep = nstep

        self._path = None if path is None else pathlib.Path(path)
        if save_frequency is not None and save_frequency <= 0:
            save_frequency = None
        self._save_frequency = save_frequency

        self._skip_truncated = skip_truncated
        self._skip_failed = skip_failed

        # if self.path is not None and self.path.exists():
        #     if num_load_entries is None:
        #         num_load_entries = capacity
        #     if num_load_entries > 0:
        #         # TODO: Support multiple workers.
        #         self.initialize()
        #         self.load(self.path, num_load_entries)

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Batch observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        """Batch action space."""
        return self._action_space

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
    def sample_strategy(self) -> SampleStrategy:
        """Sample strategy."""
        return self._sample_strategy

    @property
    def nstep(self) -> int:
        """Number of steps between sample and next observation."""
        return self._nstep

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
        self._worker_buffers = self.create_default_batch(self.worker_capacity)
        self._worker_valid_samples = np.zeros(self.worker_capacity, dtype=bool)
        self._worker_size = 0
        self._worker_idx = 0
        self._worker_idx_checkpoint = 0

    def create_default_batch(self, size: int) -> StorageBatch:
        """Creates a batch of the specified size with default values.

        Args:
            size: Batch size.

        Returns:
            Batch dict with observation, action, reward, discount, terminated,
            truncated fields.
        """
        return {
            "observation": spaces.null(self.observation_space, size),
            "action": spaces.null(self.action_space, size),
            "reward": np.full(size, float("nan"), dtype=np.float32),
            "discount": np.full(size, float("nan"), dtype=np.float32),
            "terminated": np.zeros(size, dtype=bool),
            "truncated": np.zeros(size, dtype=bool),
            "policy_args": np.empty(size, dtype=object),
        }

    def add(
        self,
        observation: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        reward: Optional[Union[np.ndarray, float]] = None,
        next_observation: Optional[np.ndarray] = None,
        discount: Optional[Union[np.ndarray, float]] = None,
        terminated: Optional[Union[np.ndarray, bool]] = None,
        truncated: Optional[Union[np.ndarray, bool]] = None,
        policy_args: Optional[Any] = None,
        batch: Optional[StorageBatch] = None,
        max_entries: Optional[int] = None,
    ) -> int:
        """Adds an experience tuple to the replay buffer.

        The experience can either be a single initial `observation`, a 5-tuple
        (`action`, `reward`, `next_observation`, `discount`, `terminated`,
        `truncated`), or a `batch` dict from buffer storage.

        The inputs can be single entries or batches.

        Args:
            observation: Initial observation.
            action: Action.
            reward: Reward.
            next_observation: Next observation.
            discount: Discount factor.
            terminated: Whether episode terminated normally.
            truncated: Whether episode terminated abnormally.
            policy_args: Auxiliary policy arguments.
            batch: Batch dict. Useful for loading from disk.
            max_entries: Limit the number of entries to add.

        Returns:
            Number of entries added.
        """
        if sum(arg is not None for arg in (observation, next_observation, batch)) != 1:
            raise ValueError(
                "Only one of observation, next_observation, or batch can be specified."
            )
        if not (
            (action is None)
            == (reward is None)
            == (next_observation is None)
            == (discount is None)
            == (terminated is None)
            == (truncated is None)
            == (policy_args is None)
        ):
            raise ValueError(
                "(action, reward, next_observation, discount, terminated, "
                "truncated, policy_args) need to be set together."
            )

        # Prepare batch.
        if observation is not None:
            dim_observation = next(nest.structure_iterator(observation)).shape
            dim_observation_space = next(
                nest.structure_iterator(self.worker_buffers["observation"])
            )
            if len(dim_observation) == len(dim_observation_space):
                batch_size = dim_observation[0]
            else:
                batch_size = 1

            batch = self.create_default_batch(batch_size)
            batch["observation"] = observation
        elif batch is None:
            assert next_observation is not None
            assert action is not None
            assert reward is not None
            assert discount is not None
            assert terminated is not None
            assert truncated is not None
            assert policy_args is not None
            batch = {
                "observation": next_observation,
                "action": action,
                "reward": reward,  # type: ignore
                "discount": discount,  # type: ignore
                "terminated": terminated,  # type: ignore
                "truncated": truncated,  # type: ignore
                "policy_args": policy_args,  # type: ignore
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

        # Compute valid samples.
        idx_batch = slice(max(0, idx_start - self.nstep), idx_stop)
        batch_terminated = _wrap_get(self.worker_buffers["terminated"], idx_batch)
        batch_truncated = _wrap_get(self.worker_buffers["truncated"], idx_batch)
        batch_discount = _wrap_get(self.worker_buffers["discount"], idx_batch)
        batch_reward = _wrap_get(self.worker_buffers["reward"], idx_batch)
        batch_done = batch_terminated | batch_truncated
        batch_valid = np.zeros_like(batch_done)

        # Ensure steps [t, t + nstep) are valid, non-terminal steps. A valid
        # index means index + nstep is a valid batch entry. Since nstep >= 1,
        # the last index is automatically invalid.
        idx_begin = np.isnan(batch_discount)
        idx_skip = idx_begin
        if self._skip_truncated:
            # Skip entire episodes if truncated.
            idx_skip |= batch_truncated
        if self._skip_failed:
            idx_skip |= (batch_reward < 1.0) & batch_done
        # done:           0 1 0 1
        # begin:          1 0 1 0
        # truncated:      0 0 0 1
        # -----------------------
        # ~done[:-1]:     1 0 1
        # ~begin[1:]:     1 0 1
        # ~truncated[1:]: 1 1 0
        # valid:          1 0 0 0
        batch_valid[:-1] = ~batch_done[:-1] & ~idx_skip[1:]
        for i in range(1, self.nstep):
            j = i + 1
            batch_valid[:-j] &= batch_done[i:-1] & ~idx_skip[j:]
        _wrap_insert(self._worker_valid_samples, batch_valid, idx_batch.start)

        # Save checkpoint.
        assert type(terminated) == type(truncated)
        if (isinstance(terminated, bool) and (terminated or truncated)) or (
            isinstance(terminated, np.ndarray) and np.any(terminated | truncated)
        ):
            len_checkpoint = self._worker_idx - self._worker_idx_checkpoint
            if (
                self.save_frequency is not None
                and len_checkpoint >= self.save_frequency
            ):
                self.save()

        return num_added

    def sample(
        self,
        sample_strategy: Optional[SampleStrategy] = None,
        batch_size: Optional[int] = None,
    ) -> Optional[Batch]:
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
        valid_samples = self._worker_valid_samples[:len_buffer]
        if sample_strategy == ReplayBuffer.SampleStrategy.SEQUENTIAL:
            idx_start = self._idx_deterministic
            num_entries = min(
                1 if batch_size is None else batch_size, len_buffer - idx_start
            )
            if num_entries <= 0:
                return None
            self._idx_deterministic += num_entries

            valid_indices = np.nonzero(valid_samples[idx_start:])[0][:num_entries]
            valid_indices[:num_entries]
            if len(valid_indices) == 0:
                return None
            elif batch_size is None:
                valid_indices = valid_indices[0]
            idx_sample = valid_indices + idx_start
        else:
            valid_indices = np.nonzero(valid_samples)[0]
            if len(valid_indices) == 0:
                return None
            idx_sample = np.random.choice(valid_indices, size=batch_size)

        # Assemble sample dict.
        observation = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample),
            self.worker_buffers["observation"],
            atom_type=np.ndarray,
        )
        next_observation = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample + self.nstep),
            self.worker_buffers["observation"],
            atom_type=np.ndarray,
        )
        action = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample + 1),
            self.worker_buffers["action"],
            atom_type=np.ndarray,
        )
        reward = np.zeros_like(self.worker_buffers["reward"][idx_sample])
        discount = np.ones_like(self.worker_buffers["discount"][idx_sample])
        for i in range(1, 1 + self.nstep):
            idx_sample_i = (idx_sample + i) % self._worker_size
            reward += discount * self.worker_buffers["reward"][idx_sample_i]
            discount *= self.worker_buffers["discount"][idx_sample_i]
        policy_args = nest.map_structure(
            functools.partial(_wrap_get, idx=idx_sample + 1),
            self.worker_buffers["policy_args"],
            atom_type=np.ndarray,
        )

        return Batch(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            discount=discount,
            policy_args=policy_args,
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
                checkpoint: StorageBatch = dict(np.load(f, allow_pickle=True))  # type: ignore
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
        len_checkpoint = idx_stop - idx_start
        if len_checkpoint == 0:
            return 0
        checkpoint = self[idx_start:idx_stop]

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

    def __iter__(self) -> Generator[Batch, None, None]:
        """Iterates over the replay buffer."""
        self.initialize()

        if self.sample_strategy == ReplayBuffer.SampleStrategy.SEQUENTIAL:
            self._idx_deterministic = 0

        while True:
            sample = self.sample()
            if sample is None:
                return
            yield sample


def _wrap_insert(
    dest: np.ndarray,
    src: Union[np.ndarray, int, float, bool],
    idx: int,
    max_entries: Optional[int] = None,
) -> int:
    """Inserts entries into the destination buffer with wrapping indices.

    Args:
        dest: Destination buffer.
        src: Source entry or batch.
        idx: Index in destination buffer to start inserting.
        max_entries: Optional maximum number of entries to insert.

    Returns:
        Number of entries added.
    """
    len_buffer = len(dest)
    idx_start = idx % len_buffer
    if isinstance(src, np.ndarray) and src.ndim == dest.ndim:
        num_entries = len(src) if max_entries is None else min(max_entries, len(src))
        idx_stop = idx_start + num_entries
        idx_split = min(len_buffer, idx_stop)
        num_added = idx_split - idx_start
        dest[idx_start:idx_split] = src[:num_added]

        if idx_split != idx_stop:
            new_max_entries = None if max_entries is None else max_entries - num_added
            return num_added + _wrap_insert(dest, src[num_added:], 0, new_max_entries)
    else:
        dest[idx_start] = src
        num_added = 1

    return num_added


def _wrap_get(
    buffer: np.ndarray,
    idx: Union[int, slice, Sequence[int]],
) -> np.ndarray:
    """Gets entries from the buffer with wrapping indices.

    Args:
        buffer: Buffer.
        idx: Numpy-style indices.

    Returns:
        Buffer slices. May be mutable views of the original buffers or temporary
        copies.
    """
    len_buffer = len(buffer)
    if isinstance(idx, int):
        return buffer[idx % len_buffer]

    if not isinstance(idx, slice):
        idx = [i % len_buffer for i in idx]
        return buffer[idx]

    # Compute number of desired entries.
    idx_start = 0 if idx.start is None else idx.start
    idx_stop = len_buffer if idx.stop is None else idx.stop
    num_entries = min(len_buffer, idx_stop - idx_start)
    if num_entries < 0:
        raise ValueError(f"Invalid slice {idx}.")

    # Wrap idx_start to range [0, len_buffer).
    idx_start = idx_start % len_buffer
    idx_stop = idx_start + num_entries
    idx_step = idx.step

    if idx_stop > len_buffer:
        len_wrap = num_entries - (len_buffer - idx_start)
        wrapped_buffer = np.concatenate(
            (buffer[idx_start:len_buffer], buffer[:len_wrap]), axis=0
        )
        return wrapped_buffer[::idx_step]
    else:
        return buffer[idx_start:idx_stop:idx_step]


if __name__ == "__main__":
    # Simple tests.
    observation_space = gym.spaces.Box(low=np.full(2, 0), high=np.full(2, 1))
    action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    replay_buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=5,
        batch_size=4,
        skip_truncated=True,
    )

    replay_buffer.initialize()
    print(replay_buffer.worker_buffers)
    print("SAMPLE", replay_buffer.sample())

    for i in range(3):
        v = 0.2 * i
        replay_buffer.add(observation=np.full(2, v))
        print("")
        print(replay_buffer.worker_buffers)
        print("SAMPLE", replay_buffer.sample())

        replay_buffer.add(
            next_observation=np.full(2, v + 0.1),
            action=np.full(1, v + 0.1),
            reward=v,
            discount=0.99,
            terminated=True,
            truncated=i == 1,
        )
        print("")
        print(replay_buffer.worker_buffers)
        print("SAMPLE", replay_buffer.sample())

    for i, batch in enumerate(replay_buffer):
        if i > 2:
            break
        print("BATCH", batch)
        print("")
    # print("SAMPLE", replay_buffer.sample())
