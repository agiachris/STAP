import collections
import time
from typing import Dict, List

import numpy as np


class Timer:
    """Timer to keep track of timing intervals for different keys."""

    def __init__(self):
        self._tics = {}

    def tic(self, key: str) -> float:
        """Starts timing for the given key.

        Args:
            key: Time interval key.

        Returns:
            Current time.
        """
        self._tics[key] = time.time()
        return self._tics[key]

    def toc(self, key: str, set_tic: bool = False) -> float:
        """Returns the time elapsed since the last tic for the given key.

        Args:
            key: Time interval key.
            set_tic: Reset the tic to the current time.

        Returns:
            Time elapsed since the last tic.
        """
        toc = time.time()
        tic = self._tics[key]
        if set_tic:
            self._tics[key] = toc
        return toc - tic


class Profiler(Timer):
    """Profiler to keep track of average time interval for different keys."""

    def __init__(self, disabled: bool = False):
        """Initializes the profiler with the given status.

        Args:
            disabled: Disable the profiler.
        """
        super().__init__()
        self._disabled = disabled
        self._tictocs: Dict[str, List[float]] = collections.defaultdict(list)

    def disable(self) -> None:
        """Disable the profiler so that tic and toc do nothing."""
        self._disabled = True

    def enable(self) -> None:
        """Enable the profiler."""
        self._disabled = False

    def tic(self, key: str) -> float:
        """Starts timing for the given key.

        Args:
            key: Time interval key.

        Returns:
            Current time.
        """
        if self._disabled:
            return 0.0
        return super().tic(key)

    def toc(self, key: str, set_tic: bool = False) -> float:
        """Returns the time elapsed since the last tic for the given key.

        Args:
            key: Time interval key.
            set_tic: Reset the tic to the current time.

        Returns:
            Time elapsed since the last tic.
        """
        if self._disabled:
            return 0.0
        tictoc = super().toc(key, set_tic)
        self._tictocs[key].append(tictoc)
        return tictoc

    def compute_average(self, key: str, reset: bool = False) -> float:
        """Computes the average time interval for the given key.

        Args:
            key: Time interval key.
            reset: Reset the collected time intervals.

        Returns:
            Average time interval.
        """
        mean = np.mean(self._tictocs[key])
        if reset:
            self._tictocs[key] = []
        return mean

    def collect_profiles(self) -> Dict[str, float]:
        """Collects and resets the average time intervals for all keys.

        Returns:
            Dict mapping from key to average time interval.
        """
        return {
            key: self.compute_average(key, reset=True) for key in self._tictocs.keys()
        }
