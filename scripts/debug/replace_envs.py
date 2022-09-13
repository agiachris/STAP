#!/usr/bin/env python3

import functools
import pathlib
import shutil
from typing import Callable

PATH = "models"
EXP_NAME = "20220912/official"
ENV_CONFIGS_PATH = "configs/pybullet/envs"
ENVS = {
    "pick": ("official/primitives/pick.yaml", "official/primitives/pick_eval.yaml"),
    "place": ("official/primitives/place.yaml", "official/primitives/place_eval.yaml"),
    "pull": ("official/primitives/pull.yaml", "official/primitives/pull_eval.yaml"),
    "push": ("official/primitives/push.yaml", "official/primitives/push_eval.yaml"),
}


def replace_envs(dry_run: bool = False) -> None:
    if dry_run:
        cp: Callable = functools.partial(print, "  - cp")
        mv: Callable = functools.partial(print, "  - mv")
    else:
        cp = shutil.copyfile
        mv = shutil.move

    env_configs_path = pathlib.Path(ENV_CONFIGS_PATH)
    for policy, (env_config, eval_env_config) in ENVS.items():
        path = pathlib.Path(PATH) / EXP_NAME / policy
        print(policy)
        mv(path / "env_config.yaml", path / "env_config.yaml.bak")
        mv(path / "eval/env_config.yaml", path / "eval/env_config.yaml.bak")
        cp(env_configs_path / env_config, path / "env_config.yaml")
        cp(env_configs_path / eval_env_config, path / "eval/env_config.yaml")


if __name__ == "__main__":
    print("Dry run...")
    replace_envs(dry_run=True)

    key = input("Copy files? (y/n): ")
    if key != "y":
        print("Aborted")
        exit()

    print("Copying...")
    replace_envs()
