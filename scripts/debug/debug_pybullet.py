#!/usr/bin/env python3

import argparse

from temporal_policies import envs
from temporal_policies.envs import pybullet
from temporal_policies.utils import timing


def main(env_config: str) -> None:
    env_factory = envs.EnvFactory(config=env_config)
    env: pybullet.table_env.TableEnv = env_factory()
    timer = timing.Timer()

    while True:
        obs = env.reset()
        action_skeleton = env.task.action_skeleton
        done = False
        for step in range(len(action_skeleton)):
            # Set and get primitive
            env.set_primitive(primitive=action_skeleton[step])
            primitive = env.get_primitive()
            input(f"Execute primitive: {primitive}")

            # Sample action and step environment
            action = primitive.sample_action()
            obs, success, _, truncated, _ = env.step(
                primitive.normalize_action(action.vector)
            )
            print(f"Success {primitive}: {success}")
            if truncated:
                break

        input("Done task, continue?\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-config",
        "-e",
        type=str,
        required=True,
        help="Path to environment config.",
    )
    args = parser.parse_args()
    main(**vars(args))
