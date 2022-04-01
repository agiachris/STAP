from typing import Iterable

import gym  # type: ignore
import numpy as np  # type: ignore


def overlay_boxes(spaces: Iterable[gym.spaces.Box]) -> gym.spaces.Box:
    """Overlays box spaces over each other.

    Args:
        spaces: Box spaces.

    Returns:
        Box space that encompasses all given spaces.
    """
    low = np.stack([space.low for space in spaces], axis=0).min(axis=0)
    high = np.stack([space.high for space in spaces], axis=0).max(axis=0)
    shape = np.stack([space.shape for space in spaces], axis=0).max(axis=0)
    return gym.spaces.Box(low=low, high=high, shape=shape)


def concatenate_boxes(spaces: Iterable[gym.spaces.Box]) -> gym.spaces.Box:
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
