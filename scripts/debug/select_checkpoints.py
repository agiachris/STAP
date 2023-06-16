#!/usr/bin/env python3

import pathlib
import shutil
import argparse


INPUT_PATH = "models"
PRIMITIVE_CHECKPOINTS = {
    "pick": {"path": "primitives_light_mse", "checkpoint": "ckpt_model_100000"},
    "place": {"path": "primitives_light_mse", "checkpoint": "ckpt_model_200000"},
    "pull": {"path": "primitives_light_mse", "checkpoint": "ckpt_model_150000"},
    "push": {"path": "primitives_light_mse", "checkpoint": "ckpt_model_150000"},
}
DYNAMICS_CHECKPOINT = {
    "path": "dynamics/pick_place_pull_push_dynamics",
    "checkpoint": "best_model",
}


def select_checkpoints(clone_name: str, clone_dynamics: bool = False) -> None:
    for primitive, info in PRIMITIVE_CHECKPOINTS.items():
        print(f"Copying {primitive}.")
        path = pathlib.Path(INPUT_PATH) / info["path"] / primitive
        ckpt_model = info["checkpoint"]
        ckpt_trainer = info["checkpoint"].replace("model", "trainer")
        shutil.copyfile(path / f"{ckpt_model}.pt", path / f"{clone_name}_model.pt")
        shutil.copyfile(path / f"{ckpt_trainer}.pt", path / f"{clone_name}_trainer.pt")

    if clone_dynamics:
        print(f"Copying dynamics.")
        path = pathlib.Path(INPUT_PATH) / DYNAMICS_CHECKPOINT["path"]
        ckpt_model = DYNAMICS_CHECKPOINT["checkpoint"]
        shutil.copyfile(path / f"{ckpt_model}.pt", path / f"{clone_name}_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-name", "-n", type=str, default="")
    parser.add_argument("--clone-dynamics", "-d", type=bool, default=False)
    args = parser.parse_args()

    select_checkpoints(args.clone_name, clone_dynamics=args.clone_dynamics)
