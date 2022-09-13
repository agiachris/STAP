#!/usr/bin/env python3

import functools
import pathlib
import shutil
from typing import Callable

PATH = "models"
EXP_NAME = "20220912/official"
CHECKPOINTS = {
    "pick": "100000",
    "place": "200000",
    "pull": "150000",
    "push": "100000",
}


def select_checkpoints(dry_run: bool = False) -> None:
    if dry_run:
        cp: Callable = functools.partial(print, "cp")
    else:
        cp = shutil.copyfile

    for policy, ckpt in CHECKPOINTS.items():
        path = pathlib.Path(PATH) / EXP_NAME / policy
        cp(path / f"ckpt_model_{ckpt}.pt", path / "select_model.pt")
        cp(path / f"ckpt_trainer_{ckpt}.pt", path / "select_trainer.pt")


if __name__ == "__main__":
    print("Dry run...")
    select_checkpoints(dry_run=True)

    key = input("Copy files? (y/n): ")
    if key != "y":
        print("Aborted")
        exit()

    print("Copying...")
    select_checkpoints()
