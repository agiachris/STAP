import argparse
import pathlib
import subprocess
from typing import Any, Dict, List

import yaml  # type: ignore
import torch  # type: ignore

import temporal_policies
from temporal_policies.algs import dynamics


def get_dynamics_model(
    train_config: Dict[str, Any],
    task_config: dynamics.TaskConfig,
    checkpoints: List[str],
    device: str = "auto",
) -> dynamics.DynamicsModel:
    dynamics_class = vars(dynamics)[train_config["dynamics"]]
    assert issubclass(dynamics_class, dynamics.DynamicsModel)
    dataset_class = (
        None
        if train_config["dataset"] is None
        else vars(temporal_policies.datasets)[train_config["dataset"]]
    )
    network_class = (
        None
        if train_config["network"] is None
        else vars(temporal_policies.networks)[train_config["network"]]
    )
    optim_class = (
        None
        if train_config["optim"] is None
        else vars(torch.optim)[train_config["optim"]]
    )
    policies = dynamics.load_policies(task_config, args.checkpoints)
    dynamics_kwargs = (
        {}
        if train_config["dynamics_kwargs"] is None
        else train_config["dynamics_kwargs"]
    )

    model = dynamics_class(
        policies,
        network_class=network_class,
        network_kwargs=train_config["network_kwargs"],
        dataset_class=dataset_class,
        dataset_kwargs=train_config["dataset_kwargs"],
        optimizer_class=optim_class,
        optimizer_kwargs=train_config["optim_kwargs"],
        **dynamics_kwargs,
    )

    return model


def save_git_hash(path: pathlib.Path):
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = process.communicate()[0].strip()
    with open(path / "git_hash.txt", "wb") as f:
        f.write(git_head_hash)


def train_dynamics(
    train_config: Dict[str, Any],
    exec_config: Dict[str, Any],
    checkpoints: List[str],
    path: pathlib.Path,
    device: str = "auto",
) -> dynamics.DynamicsModel:
    print("train_dynamics: Training dynamics model with config:")
    print(train_config)
    path.mkdir(parents=True, exist_ok=False)
    with open(path / "train_config.yaml", "w") as f:
        yaml.dump(train_config, f)
    with open(path / "exec_config.yaml", "w") as f:
        yaml.dump(exec_config, f)
    task_config = exec_config["task"]

    save_git_hash(path)

    model = get_dynamics_model(train_config, task_config, checkpoints, device)
    model.train(str(path), **train_config["train_kwargs"])
    return model


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        train_config = yaml.safe_load(f)
    with open(args.exec_config, "r") as f:
        task_config = yaml.safe_load(f)

    train_dynamics(
        train_config,
        task_config,
        args.checkpoints,
        pathlib.Path(args.path),
        args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to train config"
    )
    parser.add_argument(
        "--exec-config", type=str, required=True, help="Path to exec config"
    )
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        required=True,
        help="Path to model checkpoints",
    )
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    main(args)
