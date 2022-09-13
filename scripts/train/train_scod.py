#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Union

from temporal_policies import agents, scod, trainers
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    scod_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    model_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
    gui: Optional[int] = None,
) -> None:
    if resume:
        trainer_factory = trainers.TrainerFactory(checkpoint=path, device=device)

        print("[scripts.train.train_scod] Resuming trainer config:")
        pprint(trainer_factory.config)

        trainer = trainer_factory()
    else:
        if seed is not None:
            random.seed(seed)

        env_kwargs = {}
        if gui is not None:
            env_kwargs["gui"] = bool(gui)
        agent = agents.load(checkpoint=model_checkpoint, env_kwargs=env_kwargs)
        assert isinstance(agent, agents.RLAgent)
        scod_factory = scod.SCODFactory(
            config=scod_config,
            model=agent,
            model_checkpoint=model_checkpoint,
            env_kwargs=env_kwargs,
            device=device,
        )
        trainer_factory = trainers.TrainerFactory(
            path=path,
            config=trainer_config,
            agent=agent,
            scod=scod_factory(),
            policy_checkpoints=None if model_checkpoint is None else [model_checkpoint],
            env_kwargs=env_kwargs,
            device=device,
        )

        print("[scripts.train.train_scod] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_scod] SCOD config:")
        pprint(scod_factory.config)
        print("\n[scripts.train.train_scod] Model checkpoint:")
        pprint(model_checkpoint)
        print("")

        trainer = trainer_factory()

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        scod_factory.save_config(trainer.path)

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
    parser.add_argument("--scod-config", "-d", help="Path to SCOD config")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        help="Path to model (agents.Agent) checkpoint",
    )
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")

    main(parser.parse_args())
