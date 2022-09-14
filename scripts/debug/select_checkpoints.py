#!/usr/bin/env python3

import functools
import pathlib
import shutil
from typing import Callable
import argparse

PATH = "models"
EXP_NAME = "20220914/official"
CHECKPOINTS = {
    "pick": "100000",
    "place": "200000",
    "pull": "150000",
    "push": "150000",
}
DYNAMICS_CHECKPOINT = "200000"


def select_checkpoints(clone_name: str, dry_run: bool = False, clone_dynamics: bool = False) -> None:
    if dry_run:
        cp: Callable = functools.partial(print, "cp")
    else:
        cp = shutil.copyfile

    for policy, ckpt in CHECKPOINTS.items():
        path = pathlib.Path(PATH) / EXP_NAME / policy
        print(policy)
        cp(path / f"ckpt_model_{ckpt}.pt", path / f"select{clone_name}_model.pt")
        cp(path / f"ckpt_trainer_{ckpt}.pt", path / f"select{clone_name}_trainer.pt")
    
    if dry_run:
        cp: Callable = functools.partial(print, "cp")
    else:
        cp = shutil.copytree

    if clone_dynamics:
        path = pathlib.Path(PATH) / EXP_NAME
        print(path)
        cp(path / f"ckpt_model_{DYNAMICS_CHECKPOINT}", path / f"select{clone_name}_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-name", "-n", type=str, default="")
    parser.add_argument("--clone-dynamics", "-d", type=bool, default=False)
    args = parser.parse_args()

    print("Dry run...")
    select_checkpoints(args.clone_name, dry_run=True, clone_dynamics=args.clone_dynamics)

    key = input("Copy files? (y/n): ")
    if key != "y":
        print("Aborted")
        exit()

    print("Copying...")
    select_checkpoints(args.clone_name, clone_dynamics=args.clone_dynamics)
