from typing import Optional, Sequence, Tuple, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore


def null_tensor(
    space: gym.spaces.Space,
    batch_shape: Tuple[int, ...] = tuple(),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Constructs a null tensor from the given space.

    Args:
        space: Gym space.
        batch_shape: Batch shape.
    """

    def null_value(dtype: np.number) -> Union[float, int]:
        if dtype.kind == "f":
            return float("nan")
        else:
            return 0

    arr = np.full(
        (*batch_shape, *space.shape), null_value(space.dtype), dtype=space.dtype
    )
    arr = torch.from_numpy(arr)
    if device is not None:
        arr = arr.to(device)

    return arr


def overlay_boxes(spaces: Sequence[gym.spaces.Box]) -> gym.spaces.Box:
    """Overlays box spaces over each other.

    Args:
        spaces: Box spaces.

    Returns:
        Box space that encompasses all given spaces.
    """
    dims = max(len(space.shape) for space in spaces)

    # Compute largest shape along each axis.
    shape = np.zeros((len(spaces), dims), dtype=int)
    for idx, space in enumerate(spaces):
        shape[idx, : len(space.shape)] = space.shape
    shape = tuple(shape.max(axis=0).tolist())

    # Compute lows and highs.
    min_val = np.concatenate([space.low.flatten() for space in spaces], axis=0).min()
    max_val = np.concatenate([space.high.flatten() for space in spaces], axis=0).max()
    low = np.full((len(spaces), *shape), max_val)
    high = np.full((len(spaces), *shape), min_val)
    for idx, space in enumerate(spaces):
        low[idx, : len(space.low)] = space.low
        high[idx, : len(space.high)] = space.high
    low = low.min(axis=0)
    high = high.max(axis=0)

    return gym.spaces.Box(low=low, high=high, shape=shape)


def concatenate_boxes(spaces: Sequence[gym.spaces.Box]) -> gym.spaces.Box:
    """Concatenates box spaces.

    Args:
        spaces: Box spaces.

    Returns:
        Concatenated box space.
    """
    low = np.concatenate([space.low for space in spaces], axis=0)
    high = np.concatenate([space.high for space in spaces], axis=0)
    shape = np.array([space.shape for space in spaces]).sum(axis=0)
    return gym.spaces.Box(low=low, high=high, shape=shape)
