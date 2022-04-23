import csv
import pathlib
from typing import Any, Dict, IO, Optional, Union

import numpy as np  # type: ignore
import torch  # type: ignore


class Logger(object):
    def __init__(self, path: Union[str, pathlib.Path]):
        self.path = pathlib.Path(path)

        self._writer = torch.utils.tensorboard.SummaryWriter(log_dir=path)
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file: Optional[IO] = None

        self._staged: Dict[str, Any] = {}
        self._flushed: Dict[str, Any] = {}

    def log(
        self, key: str, value: Union[Any, Dict[str, Any]], std: bool = False
    ) -> None:
        """Logs the (key, value) pair.

        If `value` is an array, the mean and standard deviation of values will
        be logged.

        flush() must be manually called to send the results to Tensorboard.

        Args:
            key: Tensorboard key.
            value: Single value or dict of values.
            std: Whether to log the standard deviations of arrays.
        """
        if isinstance(value, np.ndarray):
            self.log(key, np.mean(value))
            self.log(f"{key}/std", np.std(value))
            return

        if isinstance(value, dict):
            for subkey, subval in value.items():
                self.log(f"{key}/{subkey}", subval)
            return

        self._staged[key] = value

    def flush(self, step: int, dump_csv: bool = False):
        """Flushes the logged values to Tensorboard.

        Args:
            step: Training step.
            dump_csv: Whether to write the log to a CSV file.
        """
        for key, value in self._staged.items():
            self._writer.add_scalar(key, value, step)
        self._writer.flush()

        self._flushed.update(self._staged)
        self._staged = {}

        if dump_csv:
            if self._csv_writer is None:
                self._csv_file = open(self.path / "log.csv", "w")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(self._staged.keys())
                )
                self._csv_writer.writeheader()
            assert self._csv_file is not None

            self._csv_writer.writerow(self._flushed)
            self._csv_file.flush()
