import csv
import pathlib
from typing import Any, Dict, IO, Optional, Union

import numpy as np
import torch
from torch.utils import tensorboard

from temporal_policies.utils import metrics


class Logger(object):
    def __init__(self, path: Union[str, pathlib.Path]):
        self.path = pathlib.Path(path)

        self._writer: Optional[tensorboard.SummaryWriter] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file: Optional[IO] = None

        self._staged: Dict[str, Any] = {}
        self._flushed: Dict[str, Any] = {}
        self._images: Dict[str, np.ndarray] = {}
        self._embeddings: Dict[str, Dict[str, torch.Tensor]] = {}

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
        subkey = key.split("/")[-1]
        if subkey.startswith("emb"):
            # Stage embedding.
            self._embeddings[key] = value
            return

        if isinstance(value, np.ndarray):
            subkey = key.split("/")[-1]

            # Stage image.
            if subkey.startswith("img"):
                self._images[key] = value
                return

            # Log mean/std of array.
            self.log(key, np.mean(value))
            if subkey in metrics.METRIC_AGGREGATION_FNS:
                self.log(f"{key}/std", np.std(value))
            return

        if isinstance(value, dict):
            for subkey, subval in value.items():
                self.log(f"{key}/{subkey}", subval)
            return

        # Stage scalar value.
        self._staged[key] = value

    def flush(self, step: int, dump_csv: bool = False):
        """Flushes the logged values to Tensorboard.

        Args:
            step: Training step.
            dump_csv: Whether to write the log to a CSV file.
        """
        if self._writer is None:
            self._writer = tensorboard.SummaryWriter(log_dir=self.path)

        for key, value in self._staged.items():
            self._writer.add_scalar(key, value, step)
        for key, img in self._images.items():
            self._writer.add_images(key, img, step)
        for key, emb in self._embeddings.items():
            self._writer.add_embedding(tag=f"{key}_{step}", **emb)
        self._writer.flush()

        self._flushed.update(self._staged)
        self._staged = {}
        self._images = {}
        self._embeddings = {}

        if dump_csv:
            if self._csv_writer is None:
                self._csv_file = open(self.path / "log.csv", "w")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(self._flushed.keys())
                )
                self._csv_writer.writeheader()
            assert self._csv_file is not None

            try:
                self._csv_writer.writerow(self._flushed)
            except ValueError:
                # Recreate csv headers.
                self._csv_file.close()
                self._csv_file = open(self.path / "log.csv", "w")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(self._flushed.keys())
                )
                self._csv_writer.writeheader()
                self._csv_writer.writerow(self._flushed)

            self._csv_file.flush()
