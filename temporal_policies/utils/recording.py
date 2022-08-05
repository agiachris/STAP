import collections
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union

import imageio
import numpy as np


class Recorder:
    def __init__(self, frequency: int = 1, max_size: Optional[int] = 1000):
        self.frequency = frequency
        self.max_size = max_size

        self._recordings: Dict[Any, List[np.ndarray]] = collections.defaultdict(list)
        self._buffer: Optional[List[np.ndarray]] = None
        self._timestep = 0

    def timestep(self) -> int:
        return self._timestep

    def is_recording(self) -> bool:
        return self._buffer is not None

    def start(
        self,
        prepend_id: Optional[str] = None,
        frequency: Optional[int] = None,
    ) -> None:
        """Starts recording.

        Existing frame buffer will be wiped out.

        Args:
            prepend_id: Upcoming recording will be prepended with the recording at this id.
            frequency: Recording frequency.
        """
        prepend_buffer = (
            [] if prepend_id is None else list(self._recordings[prepend_id])
        )
        self._buffer = prepend_buffer
        self._timestep = 0

        if frequency is not None:
            self.frequency = frequency

    def stop(self, save_id: str = "") -> bool:
        """Stops recording.

        Args:
            save_id: Saves the recording to this id.
        Returns:
            False if there is no recording to stop.
        """
        if self._buffer is None or len(self._buffer) == 0:
            return False

        self._recordings[save_id] = self._buffer
        self._buffer = None

        return True

    def save(self, path: Union[str, pathlib.Path], reset: bool = True) -> bool:
        """Saves all the recordings.

        Args:
            path: Path for the recording.
            reset: Reset the recordings after saving.
        Returns:
            False if there were no recordings to save.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        num_saved = 0
        for id, recording in self._recordings.items():
            if len(recording) == 0:
                continue

            if id is None or id == "":
                path_video = path
            else:
                path_video = path.parent / f"{path.stem}-{id}{path.suffix}"

            imageio.mimsave(path_video, recording)  # type: ignore
            num_saved += 1

        if reset:
            self._recordings.clear()

        return num_saved > 0

    def add_frame(
        self,
        grab_frame_fn: Optional[Callable[[], np.ndarray]] = None,
        frame: Optional[np.ndarray] = None,
        override_frequency: bool = False,
    ) -> bool:
        """Adds a frame to the buffer.

        Args:
            grab_frame_fn: Callback function for grabbing a frame that is only
                called if a frame is needed. Use this if rendering is expensive.
            frame: Frame to add.
            override_frequency: Add a frame regardless of the frequency.
        Returns:
            True if a frame was added.
        """
        self._timestep += 1

        if self._buffer is None:
            return False
        if self.max_size is not None and len(self._buffer) >= self.max_size:
            return False
        if not override_frequency and (self._timestep - 1) % self.frequency != 0:
            return False

        if grab_frame_fn is not None:
            frame = grab_frame_fn()
        elif frame is None:
            raise ValueError("One of grab_frame_fn or frame must not be None.")

        self._buffer.append(frame)

        return True
