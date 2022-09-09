#!/usr/bin/env python3

import argparse
from typing import Optional

from temporal_policies import envs
from temporal_policies.envs import pybullet


def main(env_config: str, seed: Optional[int] = None) -> None:
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory()
    assert isinstance(env, pybullet.table_env.TableEnv)

    while True:
        obs, info = env.reset(seed=seed)
        seed = None
        print("Reset seed:", info["seed"])

        action_skeleton = env.task.action_skeleton
        for step in range(len(action_skeleton)):
            # Set and get primitive
            env.set_primitive(primitive=action_skeleton[step])
            primitive = env.get_primitive()
            assert isinstance(primitive, pybullet.table.primitives.Primitive)
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
    parser.add_argument("--seed", "-s", type=int, help="Seed to reset env")
    args = parser.parse_args()
    main(**vars(args))
