#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Sequence, Union

from temporal_policies import encoders, envs, trainers
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    policy_checkpoints: Sequence[Union[str, pathlib.Path]],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    encoder_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
) -> None:
    if resume:
        trainer_factory = trainers.TrainerFactory(checkpoint=path, device=device)

        print("[scripts.train.train_autoencoder] Resuming trainer config:")
        pprint(trainer_factory.config)

        trainer = trainer_factory()
    else:
        if seed is not None:
            random.seed(seed)

        if env_config is None:
            env_config = envs.load_config(policy_checkpoints[0])

        env_factory = envs.EnvFactory(config=env_config)
        encoder_factory = encoders.EncoderFactory(
            config=encoder_config, env=env_factory(), device=device
        )
        trainer_factory = trainers.TrainerFactory(
            path=path,
            config=trainer_config,
            encoder=encoder_factory(),
            policy_checkpoints=policy_checkpoints,
            device=device,
        )

        print("[scripts.train.train_autoencoder] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_autoencoder] Encoder config:")
        pprint(encoder_factory.config)
        print("\n[scripts.train.train_policy] Env config:")
        pprint(env_factory.config)
        print("\n[scripts.train.train_autoencoder] Policy checkpoints:")
        pprint(policy_checkpoints)
        print("")

        trainer = trainer_factory()

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        encoder_factory.save_config(trainer.path)
        env_factory.save_config(trainer.path)

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
    parser.add_argument("--env-config", "-e", help="Path to env config")
    parser.add_argument("--encoder-config", help="Path to encoder config")
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

    main(parser.parse_args())
