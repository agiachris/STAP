#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Union

import tqdm

from temporal_policies import agents, dynamics, envs, planners, trainers
from temporal_policies.utils import configs, random


def train(
    path: Union[str, pathlib.Path],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    dynamics_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    agent_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    eval_env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    planner_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    eval_recording_path: Optional[Union[str, pathlib.Path]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
    gui: Optional[int] = None,
    num_pretrain_steps: Optional[int] = None,
    num_train_steps: Optional[int] = None,
    num_eval_steps: Optional[int] = None,
    num_env_processes: Optional[int] = None,
    num_eval_env_processes: Optional[int] = None,
    closed_loop_planning: int = 1,
) -> None:
    if resume:
        trainer_factory = trainers.TrainerFactory(checkpoint=path, device=device)

        print("[scripts.train.train_policy] Resuming trainer config:")
        pprint(trainer_factory.config)

        trainer = trainer_factory()
    else:
        if seed is not None:
            random.seed(seed)

        if agent_config is None:
            raise ValueError("agent_config must be specified")
        if env_config is None:
            raise ValueError("env_config must be specified")
        if planner_config is None:
            raise ValueError("planner_config must be specified")

        env_kwargs: Dict[str, Any] = {}
        eval_env_kwargs: Dict[str, Any] = {}
        if gui is not None:
            env_kwargs["gui"] = bool(gui)
            eval_env_kwargs["gui"] = bool(gui)
        if num_env_processes is not None:
            env_kwargs["num_processes"] = num_env_processes
            eval_env_kwargs["num_processes"] = num_eval_env_processes
        env_factory = envs.EnvFactory(config=env_config)
        env = env_factory(**env_kwargs)
        if eval_env_config is not None:
            eval_env_factory = envs.EnvFactory(config=eval_env_config)
            eval_env = eval_env_factory(**eval_env_kwargs)

        agent_factories = [
            agents.AgentFactory(config=agent_config, env=env, device=device)
            for _ in env.primitives
        ]

        # Assign all policies to the same encoder.
        policies = [agent_factory() for agent_factory in agent_factories]
        for policy in policies[1:]:
            del policy._encoder
            policy._encoder = policies[0].encoder

        dynamics_factory = dynamics.DynamicsFactory(
            config=dynamics_config,
            env=env,
            policies=policies,
            device=device,
        )
        dynamics_model = dynamics_factory()
        planner_factory = planners.PlannerFactory(
            config=planner_config,
            env=env,
            policies=dynamics_model.policies,
            dynamics=dynamics_model,
        )
        trainer_factory = trainers.TrainerFactory(
            path=path,
            config=trainer_config,
            dynamics=dynamics_model,
            eval_env=None if eval_env_config is None else eval_env,
            device=device,
        )

        trainer_kwargs = {
            "env": env,
            "planner": planner_factory(),
            "closed_loop_planning": bool(closed_loop_planning),
        }
        if num_pretrain_steps is not None:
            trainer_kwargs["num_pretrain_steps"] = num_pretrain_steps
        if num_train_steps is not None:
            trainer_kwargs["num_train_steps"] = num_train_steps
        if num_eval_steps is not None:
            trainer_kwargs["num_eval_steps"] = num_eval_steps

        print("[scripts.train.train_baseline] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_baseline] Dynamics config:")
        pprint(dynamics_factory.config)
        for agent_factory in agent_factories:
            print("\n[scripts.train.train_baseline] Agent config:")
            pprint(agent_factory.config)
        print("\n[scripts.train.train_baseline] Env config:")
        pprint(env_factory.config)
        if eval_env_config is not None:
            print("\n[scripts.train.train_baseline] Eval env config:")
            pprint(eval_env_factory.config)
        print("\n[scripts.train.train_baseline] Planner config:")
        pprint(planner_factory.config)
        print("")

        trainer = trainer_factory(**trainer_kwargs)

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        for agent_trainer in trainer.agent_trainers:
            agent_trainer.path.mkdir(parents=True, exist_ok=overwrite)
        trainer.dynamics_trainer.path.mkdir(parents=True, exist_ok=overwrite)

        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        dynamics_factory.save_config(trainer.dynamics_trainer.path)
        env_factory.save_config(trainer.path)
        for agent_factory, agent_trainer in zip(
            agent_factories, trainer.agent_trainers
        ):
            agent_factory.save_config(agent_trainer.path)
            env_factory.save_config(agent_trainer.path)

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
            trainer.env.record_start()
            trainer.evaluate_step()
            trainer.env.record_stop()
            trainer.env.record_save(
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
    parser.add_argument("--dynamics-config", "-d", help="Path to dynamics config")
    parser.add_argument("--agent-config", "-a", help="Paths to agent config")
    parser.add_argument("--env-config", "-e", help="Path to env config")
    parser.add_argument("--eval-env-config", help="Path to eval env config")
    parser.add_argument("--planner-config", help="Path to planner config")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--eval-recording-path")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument(
        "--num-pretrain-steps", type=int, help="Number of steps to pretrain"
    )
    parser.add_argument("--num-train-steps", type=int, help="Number of steps to train")
    parser.add_argument(
        "--num-eval-steps", type=int, help="Number of steps per evaluation"
    )
    parser.add_argument("--num-env-processes", type=int, help="Number of env processes")
    parser.add_argument(
        "--num-eval-env-processes", type=int, help="Number of eval env processes"
    )
    parser.add_argument(
        "--closed-loop-planning", type=int, default=1, help="Use closed-loop planning"
    )

    main(parser.parse_args())
