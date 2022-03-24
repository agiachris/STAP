import argparse
import pathlib
import subprocess
from typing import Any, Dict, List

import yaml  # type: ignore

from temporal_policies.algs import dynamics
from temporal_policies.utils import dynamics as dynamics_utils


def save_git_hash(path: pathlib.Path):
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = process.communicate()[0].strip()
    with open(path / "git_hash.txt", "wb") as f:
        f.write(git_head_hash)


def train_dynamics(
    dynamics_config: Dict[str, Any],
    exec_config: Dict[str, Any],
    policy_checkpoints: List[str],
    path: pathlib.Path,
    device: str = "auto",
) -> dynamics.DynamicsModel:
    print("train_dynamics: Training dynamics model with config:")
    print(dynamics_config)
    path.mkdir(parents=True, exist_ok=False)
    with open(path / "dynamics_config.yaml", "w") as f:
        yaml.dump(dynamics_config, f)
    with open(path / "exec_config.yaml", "w") as f:
        yaml.dump(exec_config, f)
    task_config = exec_config["task"]

    save_git_hash(path)

    model = dynamics_utils.load_dynamics_model(dynamics_config, task_config, policy_checkpoints, device)
    model.train(str(path), **dynamics_config["train_kwargs"])
    return model


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        dynamics_config = yaml.safe_load(f)
    with open(args.exec_config, "r") as f:
        task_config = yaml.safe_load(f)

    train_dynamics(
        dynamics_config,
        task_config,
        args.checkpoints,
        pathlib.Path(args.path),
        args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to dynamics config"
    )
    parser.add_argument(
        "--exec-config", type=str, required=True, help="Path to exec config"
    )
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument(
        "--policy-checkpoints",
        nargs="+",
        type=str,
        required=True,
        help="Path to policy checkpoints",
    )
    parser.add_argument("--device", "-d", type=str, default="auto")

    main(parser.parse_args())
