#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Union

from temporal_policies import agents, envs, trainers
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    agent_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    encoder_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
) -> None:
    if resume:
        trainer_factory = trainers.TrainerFactory(checkpoint=path, device=device)

        print("[scripts.train.train_policy] Resuming trainer config:")
        pprint(trainer_factory.config)

        trainer = trainer_factory()
    else:
        if seed is not None:
            random.seed(seed)

        if env_config is None:
            raise ValueError("env_config must be specified")

        env_factory = envs.EnvFactory(config=env_config)
        agent_factory = agents.AgentFactory(
            config=agent_config,
            env=env_factory(),
            encoder_checkpoint=encoder_checkpoint,
            device=device,
        )
        trainer_factory = trainers.TrainerFactory(
            path=path, config=trainer_config, agent=agent_factory(), device=device
        )

        print("[scripts.train.train_policy] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_policy] Agent config:")
        pprint(agent_factory.config)
        print("\n[scripts.train.train_policy] Env config:")
        pprint(env_factory.config)
        print("")

        trainer = trainer_factory()

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        agent_factory.save_config(trainer.path)
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
    parser.add_argument(
        "--agent-config", "-a", required=True, help="Path to agent config"
    )
    parser.add_argument("--env-config", "-e", required=True, help="Path to env config")
    parser.add_argument("--encoder-checkpoint", help="Path to encoder checkpoint")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")

    main(parser.parse_args())
