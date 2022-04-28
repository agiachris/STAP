#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Sequence, Union

from temporal_policies import agents, dynamics, envs, trainers
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    dynamics_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    agent_configs: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any]]]] = None,
    env_configs: Optional[Sequence[Union[str, pathlib.Path, Dict[str, Any]]]] = None,
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

        if agent_configs is None:
            raise ValueError("agent_configs must be specified")
        if env_configs is None:
            raise ValueError("env_configs must be specified")

        env_factories = [
            envs.EnvFactory(config=env_config) for env_config in env_configs
        ]
        agent_factories = [
            agents.AgentFactory(config=agent_config, env=env_factory(), device=device)
            for agent_config, env_factory in zip(agent_configs, env_factories)
        ]

        # Assign all policies to the same encoder.
        policies = [agent_factory() for agent_factory in agent_factories]
        for policy in policies[1:]:
            del policy._encoder
            policy._encoder = policies[0].encoder

        dynamics_factory = dynamics.DynamicsFactory(
            config=dynamics_config,
            policies=policies,
            device=device,
        )
        trainer_factory = trainers.TrainerFactory(
            path=path, config=trainer_config, dynamics=dynamics_factory(), device=device
        )

        print("[scripts.train.train_unified] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_unified] Dynamics config:")
        pprint(dynamics_factory.config)
        for agent_factory in agent_factories:
            print("\n[scripts.train.train_unified] Agent config:")
            pprint(agent_factory.config)
        for env_factory in env_factories:
            print("\n[scripts.train.train_unified] Env config:")
            pprint(env_factory.config)
        print("")

        trainer = trainer_factory()

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        for agent_trainer in trainer.agent_trainers:
            agent_trainer.path.mkdir(parents=True, exist_ok=overwrite)
        trainer.dynamics_trainer.path.mkdir(parents=True, exist_ok=overwrite)

        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        dynamics_factory.save_config(trainer.dynamics_trainer.path)
        for agent_factory, env_factory, agent_trainer in zip(
            agent_factories, env_factories, trainer.agent_trainers
        ):
            agent_factory.save_config(agent_trainer.path)
            env_factory.save_config(agent_trainer.path)

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
        "--agent-configs", "-a", nargs="+", help="Paths to agent configs"
    )
    parser.add_argument("--env-configs", "-e", nargs="+", help="Paths to env config")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")

    main(parser.parse_args())
