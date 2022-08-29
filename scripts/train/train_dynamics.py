#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Sequence, Union

from temporal_policies import trainers
from temporal_policies.dynamics import DynamicsFactory
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    dynamics_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
    gui: Optional[int] = None,
    num_train_steps: Optional[int] = None,
) -> None:
    if resume:
        trainer_factory = trainers.TrainerFactory(checkpoint=path, device=device)

        print("[scripts.train.train_dynamics] Resuming trainer config:")
        pprint(trainer_factory.config)

        trainer = trainer_factory()
    else:
        if seed is not None:
            random.seed(seed)

        env_kwargs = {}
        if gui is not None:
            env_kwargs["gui"] = bool(gui)
        dynamics_factory = DynamicsFactory(
            config=dynamics_config,
            policy_checkpoints=policy_checkpoints,
            env_kwargs=env_kwargs,
            device=device,
        )
        dynamics = dynamics_factory()
        trainer_factory = trainers.TrainerFactory(
            path=path,
            config=trainer_config,
            dynamics=dynamics,
            policy_checkpoints=policy_checkpoints,
            policies=dynamics.policies,
            env_kwargs=env_kwargs,
            device=device,
        )
        trainer_kwargs = {}
        if num_train_steps is not None:
            trainer_kwargs["num_train_steps"] = num_train_steps

        print("[scripts.train.train_dynamics] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_dynamics] Dynamics config:")
        pprint(dynamics_factory.config)
        print("\n[scripts.train.train_dynamics] Policy checkpoints:")
        pprint(policy_checkpoints)
        print("")

        trainer = trainer_factory(**trainer_kwargs)

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        dynamics_factory.save_config(trainer.path)

    trainer.train()


def main(args: argparse.Namespace) -> None:
    train(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer-config",
        "--config",
        "-c",
        required=True,
        help="Path to trainer config",
    )
    parser.add_argument("--dynamics-config", "-d", help="Path to dynamics config")
    parser.add_argument(
        "--policy-checkpoints",
        nargs="+",
        type=str,
        help="Path to policy checkpoints",
    )
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--num-train-steps", type=int, help="Number of steps to train")

    main(parser.parse_args())
