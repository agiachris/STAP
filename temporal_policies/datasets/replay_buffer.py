import datetime
import enum
import functools
import pathlib
from typing import Generator, Optional, Sequence, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies.utils import nest

Batch = nest.NestedStructure


class ReplayBuffer(torch.utils.data.IterableDataset):
    """Replay buffer class."""

    class SampleStrategy(enum.Enum):
        """Replay buffer sample strategy."""

        UNIFORM = 0  # Uniform random sampling.
        SEQUENTIAL = 1  # Deterministic sequential order.

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        capacity: int = 100000,
        batch_size: Optional[int] = None,
        sample_strategy: Union[str, SampleStrategy] = "uniform",
        nstep: int = 1,
        path: Optional[Union[str, pathlib.Path]] = None,
        save_frequency: Optional[int] = None,
    ):
        """Stores the configuration parameters for the replay buffer.

        The actual buffers will be constructed upon calling
        `ReplayBuffer.__iter__() or `ReplayBuffer.initialize()`.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            capacity: Replay buffer capacity.
            batch_size: Sample batch size.
            sample_strategy: Sample strategy.
            nstep: Number of steps between sample and next observation.
            path: Optional location of replay buffer on disk.
            save_frequency: Frequency of optional automatic saving to disk.
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

    @property
    def capacity(self) -> int:
        """Replay buffer capacity."""
        return self._capacity

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
        """Initializes the buffers."""

        def create_buffer(space: gym.spaces.Space, capacity: int):
            if isinstance(space, gym.spaces.Discrete):
                return np.full(capacity, space.start - 1, dtype=np.int64)
            elif isinstance(space, gym.spaces.Box):
                return np.full(
                    (capacity, *space.shape), float("nan"), dtype=space.dtype
                )
            elif isinstance(space, gym.spaces.Tuple):
                return tuple(create_buffer(s, capacity) for s in space)
            elif isinstance(space, gym.spaces.Dict):
                return {key: create_buffer(s, capacity) for key, s in space.items()}
            else:
                raise ValueError("Invalid space provided")

        # Set up only once.
        if hasattr(self, "_worker_buffers"):
            return

        # TODO: Need to think about how to load data among multiple workers when
        # multiple policies are being trained.
        if self.num_workers != 1:
            raise NotImplementedError("Multiple workers not supported yet.")

        self._worker_capacity = self.capacity // self.num_workers
        self._worker_buffers = {
            "observation": create_buffer(self._observation_space, self.worker_capacity),
            "action": create_buffer(self._action_space, self.worker_capacity),
            "reward": np.full(self.worker_capacity, float("nan"), dtype=np.float32),
            "discount": np.full(self.worker_capacity, float("nan"), dtype=np.float32),
            "done": np.zeros(self.worker_capacity, dtype=bool),
        }
        self._worker_size = 0
        self._worker_idx = 0
        self._worker_idx_checkpoint = 0

    def add(
        self,
        observation: Optional[Batch] = None,
        action: Optional[Batch] = None,
        reward: Optional[Union[np.ndarray, float]] = None,
        next_observation: Optional[Batch] = None,
        discount: Optional[Union[np.ndarray, float]] = None,
        done: Optional[Union[np.ndarray, bool]] = None,
        batch: Optional[Batch] = None,
        max_entries: Optional[int] = None,
    ) -> int:
        """Adds an experience tuple to the replay buffer.

        The experience can either be a single initial `observation`, a 5-tuple
        (`action`, `reward`, `next_observation`, `discount`, `done`), or a
        `batch` dict from buffer storage.

        The inputs can be single entries or batches.

        Args:
            observation: Initial observation.
            action: Action.
            reward: Reward.
            next_observation: Next observation.
            discount: Discount factor.
            done: Whether episode is done.
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
            == (done is None)
        ):
            raise ValueError(
                "(action, reward, next_observation, discount, done) need to be set together."
            )

        if observation is not None:
            buffers = self.worker_buffers["observation"]
            batch = observation
        elif batch is None:
            buffers = self.worker_buffers
            batch = {
                "observation": next_observation,
                "action": action,
                "reward": reward,
                "discount": discount,
                "done": done,
            }
        else:
            buffers = self.worker_buffers

        idx_start = self._worker_idx
        num_added_structure = nest.map_structure(
            functools.partial(_wrap_insert, idx=idx_start, max_entries=max_entries),
            buffers,
            batch,
            atom_type=np.ndarray,
        )
        num_added = next(nest.structure_iterator(num_added_structure, atom_type=int))
        idx_stop = idx_start + num_added

        self._worker_idx = idx_stop
        self._worker_size = min(self.worker_capacity, idx_stop)

        if not isinstance(done, bool) or done:
            len_checkpoint = self._worker_idx - self._worker_idx_checkpoint
            if (
                self.save_frequency is not None
                and len_checkpoint >= self.save_frequency
            ):
                self.save()

        return num_added

    def sample(
        self, sample_strategy: Optional[SampleStrategy] = None
    ) -> Optional[Batch]:
        """Samples a batch from the replay buffer.

        Args:
            sample_strategy: Optional sample strategy.

        Returns:
            Sample batch.
        """
        if sample_strategy is None:
            sample_strategy = self.sample_strategy

        # Ensure steps [t, t + nstep) aren't terminal steps.
        len_buffer = self._worker_size
        is_valid = np.roll(
            self.worker_buffers["discount"] == self.worker_buffers["discount"], -1
        )[:len_buffer]
        is_not_terminal = ~self.worker_buffers["done"][:len_buffer]
        # TODO: Bug with incomplete episodes due to wrapping.
        for i in range(self.nstep):
            is_valid &= np.roll(is_not_terminal, -i)

        # Get sample indices.
        batch_size = 1 if self.batch_size is None else self.batch_size
        if sample_strategy == ReplayBuffer.SampleStrategy.SEQUENTIAL:
            idx_start = self._idx_deterministic
            num_entries = min(batch_size, len_buffer - idx_start)
            if num_entries <= 0:
                return None
            self._idx_deterministic += num_entries

            is_valid = is_valid[idx_start:]
            idx_sample = idx_start + np.nonzero(is_valid)[0][:num_entries]
            if len(idx_sample) == 0:
                return None
        else:
            valid_indices = np.nonzero(is_valid)[0]
            if len(valid_indices) == 0:
                return {}
            idx_sample = np.random.choice(valid_indices, size=batch_size)

        if self.batch_size is None:
            idx_sample = np.squeeze(idx_sample)

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

        return {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "discount": discount,
        }

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

        num_loaded = 0
        checkpoint_paths = sorted(
            path.iterdir(), key=lambda f: tuple(map(int, f.stem.split("_")[:-1]))
        )
        for checkpoint_path in tqdm.tqdm(checkpoint_paths):
            with open(checkpoint_path, "rb") as f:
                checkpoint = dict(np.load(f))
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

    if num_entries > len_buffer:
        len_wrap = num_entries - (len_buffer - idx_start)
        wrapped_buffer = np.concatenate(
            buffer[idx_start:len_buffer], buffer[:len_wrap], axis=0
        )
        return wrapped_buffer[::idx_step]
    else:
        return buffer[idx_start:idx_stop:idx_step]
