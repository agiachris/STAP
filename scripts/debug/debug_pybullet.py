#!/usr/bin/env python3

import argparse

from temporal_policies import envs
from temporal_policies.envs import pybullet
from temporal_policies.utils import timing


def main(env_config: str) -> None:
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory()
    timer = timing.Timer()

    while True:
        timer.tic("reset")
        obs, info = env.reset()
        dt_reset = timer.toc("reset")

        primitive = env.get_primitive()
        assert isinstance(primitive, pybullet.table.primitives.Primitive)

        print(primitive)
        print("obs:", obs)
        input("continue?")

        timer.tic("step")
        action = primitive.sample_action()
        obs, success, _, _, _ = env.step(primitive.normalize_action(action.vector))
        dt_step = timer.toc("step")

        print(f"SUCCESS {primitive}:", success, ", time:", dt_reset + dt_step)
        input("continue?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", "-e", type=str, required=True, help="Path to environment config.")
    args = parser.parse_args()
    main(**vars(args))
