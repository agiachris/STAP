import argparse
import pathlib

import yaml  # type: ignore

from temporal_policies import dynamics


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        dynamics_config = yaml.safe_load(f)
    with open(args.exec_config, "r") as f:
        task_config = yaml.safe_load(f)

    dynamics.train_dynamics(
        dynamics_config,
        task_config,
        args.policy_checkpoints,
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
