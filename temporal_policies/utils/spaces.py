from typing import Optional, Sequence, Tuple, Union

import gym
import numpy as np
import torch


def null(space: gym.spaces.Space, batch_shape: Union[int, Sequence[int]] = tuple()):
    """Constructs a nan or zero array from the given space.

    Args:
        space: Gym space.
        batch_shape: Batch shape.

    Returns:
        A nan or zero array with shape (*batch_shape, *space.shape).
    """

    def null_value(dtype: np.number) -> Union[float, int]:
        if dtype.kind == "f":
            return float("nan")
        else:
            return 0

    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)

    if isinstance(space, gym.spaces.Discrete):
        return np.full(*batch_shape, space.start - 1, dtype=np.int64)
    elif isinstance(space, gym.spaces.Box):
        return np.full(
            (*batch_shape, *space.shape), null_value(space.dtype), dtype=space.dtype
        )
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(null(s, *batch_shape) for s in space)
    elif isinstance(space, gym.spaces.Dict):
        return {key: null(s, *batch_shape) for key, s in space.items()}
    else:
        raise ValueError("Invalid space provided")


def null_tensor(
    space: gym.spaces.Space,
    batch_shape: Union[int, Sequence[int]] = tuple(),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Constructs a null tensor from the given space.

    Args:
        space: Gym space.
        batch_shape: Batch shape.

    Returns:
        A nan or zero tensor with shape (*batch_shape, *space.shape).
    """
    arr = torch.from_numpy(null(space, batch_shape))
    if device is not None:
        arr = arr.to(device)

    return arr


def batch_shape(space: gym.spaces.Box, x: np.ndarray) -> Tuple[int, ...]:
    """Computes the batch shape of the array using the given space.

    Args:
        space: Space to which the array belongs.
        x: Array.

    Returns:
        Batch shape.
    """
    return x.shape[: -len(space.shape)]


def pad_null(x: np.ndarray, space: gym.spaces.Box) -> np.ndarray:
    """Pads the array with nan or zero values to fit the space shape.

    Args:
        x: Array that is a subset of the given space.
        space: Gym space.

    Returns:
        Nan-padded array of shape `space.shape`.
    """
    shape = batch_shape(space, x)
    padded_x = null(space, shape)
    if not shape:
        padded_x[: x.shape[0]] = x
    elif len(shape) == 1:
        padded_x[:, : x.shape[1]] = x
    else:
        raise NotImplementedError
    return padded_x


def subspace(x: np.ndarray, space: gym.spaces.Box) -> np.ndarray:
    """Extracts the subspace of x.

    Args:
        space: Gym space to extract.
        x: Array that is a superset of the given space.

    Returns:
        Slice of x with shape `space.shape`.
    """
    if len(space.shape) != 1:
        raise NotImplementedError

    shape = batch_shape(space, x)

    if not shape:
        return x[: space.shape[0]]
    elif len(shape) == 1:
        return x[:, : space.shape[0]]
    else:
        raise NotImplementedError


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


def normalize(x: np.ndarray, space: gym.spaces.Box) -> np.ndarray:
    """Normalizes the space element to values between [0, 1].

    Args:
        x: Array to scale.
        space: Gym space.

    Returns:
        Array with values between [0, 1].
    """
    assert (space.low <= x).all() and (x <= space.high).all()
    return (x - space.low) / (space.high - space.low)


def unnormalize(x: np.ndarray, space: gym.spaces.Box) -> np.ndarray:
    """Unnormalizes the normalized array to values of the space.

    Args:
        x: Normalized array with values between [0, 1].
        space: Gym space.

    Returns:
        Space element.
    """
    assert (0 <= x).all() and (x <= 1).all()
    return (space.high - space.low) * x + space.low


def transform(
    x: np.ndarray, from_space: gym.spaces.Box, to_space: gym.spaces.Box
) -> np.ndarray:
    """Transforms the array from one space to another.

    Args:
        x: Element of `from_space`.
        from_space: Gym space to transform from.
        to_space: Gym space to transform to.

    Returns:
        Element of `to_space`.
    """
    return unnormalize(normalize(x, from_space), to_space)
