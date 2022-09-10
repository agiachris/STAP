import pathlib
from typing import Any, Generator, Optional, Sequence, Union

import numpy as np

from temporal_policies.datasets.replay_buffer import ReplayBuffer, StorageBatch
from temporal_policies.utils import tensors, spaces
from temporal_policies.utils.typing import Batch, WrappedBatch


class StratifiedReplayBuffer(ReplayBuffer):
    """Stratified replay buffer class.

    Used by the dynamics class to construct a batch with an equal number of
    samples from a set of child replay buffers.
    """

    def __init__(
        self,
        replay_buffers: Sequence[ReplayBuffer],
        batch_size: Optional[int] = None,
    ):
        """Connects to the child replay buffers.

        The actual buffers will be constructed upon calling
        `StratifiedReplayBuffer.__iter__() or `StratifiedReplayBuffer.initialize()`.

        Args:
            replay_buffers: Child replay buffers.
            batch_size: Sample batch size, which does not have to be a perfect
                multiple of the number of child replay buffers.
        """
        if any(
            rb._observation_space != replay_buffers[0]._observation_space
            for rb in replay_buffers[1:]
        ):
            raise ValueError("Replay buffers must have the same observation spaces")
        self._observation_space = replay_buffers[0]._observation_space
        self._action_space = spaces.overlay_boxes(
            [rb._action_space for rb in replay_buffers]
        )
        self._capacity = sum(rb.capacity for rb in replay_buffers)

        self._batch_size = batch_size
        self._sample_strategy = ReplayBuffer.SampleStrategy.UNIFORM
        self._nstep = 1

        self._replay_buffers = replay_buffers

    @property
    def replay_buffers(self) -> Sequence[ReplayBuffer]:
        """Child replay buffers."""
        return self._replay_buffers

    @property
    def num_buffers(self) -> int:
        """Number of child replay buffers."""
        return len(self.replay_buffers)

    def __len__(self) -> int:
        """Total number of entries added to the child replay buffers."""
        return sum(len(replay_buffer) for replay_buffer in self.replay_buffers)

    @property
    def path(self) -> Optional[pathlib.Path]:
        """Stratified replay buffer has no path."""
        raise NotImplementedError

    @path.setter
    def path(self, path: Union[str, pathlib.Path]) -> None:
        """Stratified replay buffer has no path."""
        raise NotImplementedError

    @property
    def save_frequency(self) -> Optional[int]:
        """Stratified replay buffer cannot be saved to disk."""
        raise NotImplementedError

    def initialize(self) -> None:
        """Initializes the worker buffers."""

        for rb in self.replay_buffers:
            rb.initialize()

    def add(
        self,
        observation: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        reward: Optional[Union[np.ndarray, float]] = None,
        next_observation: Optional[np.ndarray] = None,
        discount: Optional[Union[np.ndarray, float]] = None,
        terminated: Optional[Union[np.ndarray, bool]] = None,
        truncated: Optional[Union[np.ndarray, bool]] = None,
        policy_args: Optional[Union[np.ndarray, Any]] = None,
        batch: Optional[StorageBatch] = None,
        max_entries: Optional[int] = None,
    ) -> int:
        """Stratified replay buffer cannot be modified."""
        raise NotImplementedError

    def sample(
        self,
        sample_strategy: Optional[ReplayBuffer.SampleStrategy] = None,
        batch_size: Optional[int] = None,
    ) -> Optional[WrappedBatch]:
        """Samples a batch from the replay buffer.

        An equal number of samples are taken from the child replay buffers, with
        some randomly sampled extras to fill up the required batch size.

        Adds the `idx_replay_buffer` key to indicate the index of the child
        replay buffer that each sample in the batch comes from.

        Args:
            sample_strategy: Optional sample strategy.
            batch_size: Optional batch size. Otherwise uses default batch size.

        Returns:
            Sample batch.
        """
        if batch_size is None:
            batch_size = self.batch_size
        assert isinstance(batch_size, int)

        buffer_batch_sizes = np.full(
            self.num_buffers, int(np.ceil(batch_size / self.num_buffers) + 0.5)
        )
        num_extras = buffer_batch_sizes.sum() - batch_size
        idx_extras = np.random.choice(self.num_buffers, num_extras, replace=False)
        buffer_batch_sizes[idx_extras] -= 1
        assert buffer_batch_sizes.sum() == batch_size

        batches = [
            rb.sample(sample_strategy, buffer_batch_size)
            for (rb, buffer_batch_size) in zip(self.replay_buffers, buffer_batch_sizes)
        ]
        stratified_batches = []
        for idx_replay_buffer, batch in enumerate(batches):
            if batch is None:
                print(
                    "[temporal_policies.datasets.StratifiedReplayBuffer.sample]: "
                    f"WARNING: Batch {idx_replay_buffer}/{len(self.replay_buffers)} is empty."
                )
                continue
            assert isinstance(batch["action"], np.ndarray)
            stratified_batch = WrappedBatch(
                observation=batch["observation"],
                action=spaces.pad_null(batch["action"], self.action_space),
                reward=batch["reward"],
                next_observation=batch["next_observation"],
                discount=batch["discount"],
                policy_args=batch["policy_args"],
                idx_replay_buffer=np.full_like(
                    batch["discount"], idx_replay_buffer, dtype=int
                ),
            )
            stratified_batches.append(stratified_batch)
        if len(stratified_batches) == 0:
            return None
        stratified_batch = tensors.map_structure(
            lambda *xs: np.concatenate(xs, axis=0), *stratified_batches
        )

        return stratified_batch

    def load(
        self, path: Optional[pathlib.Path] = None, max_entries: Optional[int] = None
    ) -> int:
        """Stratified replay buffer cannot be loaded from disk."""
        raise NotImplementedError

    def save(self, path: Optional[pathlib.Path] = None) -> int:
        """Stratified replay buffer cannot be saved to disk."""
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]):
        """Stratified replay buffer cannot be indexed."""
        raise NotImplementedError

    def __iter__(self) -> Generator[Batch, None, None]:
        """Iterates over the replay buffer."""
        self.initialize()

        while True:
            sample = self.sample()
            if sample is None:
                return
            yield sample
