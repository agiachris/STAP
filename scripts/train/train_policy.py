#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Union

import tqdm

from temporal_policies import agents, envs, trainers
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    agent_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    eval_env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    encoder_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    eval_recording_path: Optional[Union[str, pathlib.Path]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
    gui: Optional[int] = None,
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
        eval_env_factory = (
            None if eval_env_config is None else envs.EnvFactory(config=eval_env_config)
        )
        env_kwargs = {}
        if gui is not None:
            env_kwargs["gui"] = bool(gui)
        env = env_factory(**env_kwargs)
        eval_env = None if eval_env_factory is None else eval_env_factory(**env_kwargs)

        agent_factory = agents.AgentFactory(
            config=agent_config,
            env=env,
            encoder_checkpoint=encoder_checkpoint,
            device=device,
        )
        trainer_factory = trainers.TrainerFactory(
            path=path,
            config=trainer_config,
            agent=agent_factory(),
            eval_env=eval_env,
            device=device,
        )

        print("[scripts.train.train_policy] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_policy] Agent config:")
        pprint(agent_factory.config)
        print("\n[scripts.train.train_policy] Env config:")
        pprint(env_factory.config)
        if eval_env_factory is not None:
            print("\n[scripts.train.train_policy] Eval env config:")
            pprint(eval_env_factory.config)
        print("")

        trainer = trainer_factory()

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        agent_factory.save_config(trainer.path)
        env_factory.save_config(trainer.path)
        if eval_env_factory is not None:
            eval_path = trainer.path / "eval"
            eval_path.mkdir(parents=True, exist_ok=overwrite)
            eval_env_factory.save_config(eval_path)

    trainer.train()

    # Record gifs of the trained policy.
    if eval_recording_path is not None:
        eval_recording_path = pathlib.Path(eval_recording_path)

        trainer.eval_mode()
        pbar = tqdm.tqdm(
            range(trainer.num_eval_steps),
            desc=f"Record {trainer.name}",
            dynamic_ncols=True,
        )
        for i in pbar:
            trainer.eval_env.record_start()
            trainer.evaluate_step()
            trainer.eval_env.record_stop()
            trainer.eval_env.record_save(
                eval_recording_path / trainer.env.name / f"eval_{i}.gif", reset=True
            )


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
    parser.add_argument("--eval-env-config", help="Path to evaluation env config")
    parser.add_argument("--encoder-checkpoint", help="Path to encoder checkpoint")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--eval-recording-path")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")

    main(parser.parse_args())
