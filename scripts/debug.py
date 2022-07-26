#!/usr/bin/env python3

from temporal_policies import envs
from temporal_policies.envs import pybullet
from temporal_policies.utils import timing


def main() -> None:
    env_factory = envs.EnvFactory(config="configs/pybullet/envs/pick.yaml")
    env = env_factory()
    assert isinstance(env, pybullet.TableEnv)
    primitive = env.get_primitive()
    assert isinstance(primitive, pybullet.table.primitives.Primitive)

    timer = timing.Timer()
    while True:
        timer.tic("reset")
        obs = env.reset()
        print(env.object_states())
        print("obs:", obs)
        input("continue?")
        dt_reset = timer.toc("reset")

        timer.tic("step")
        action = primitive.sample_action()
        obs, success, _, _ = env.step(primitive.normalize_action(action))
        input("continue?")
        dt_step = timer.toc("step")

        print(f"SUCCESS {env.get_primitive()}:", success, ", time:", dt_reset + dt_step)
        if not success:
            continue

        # env.set_primitive(env.action_skeleton[1])
        # obs = env.get_observation()
        #
        # xyz_min, xyz_max = env.primitive.args[1].aabb()
        # xyz = np.zeros(3)
        # xyz[:2] = np.random.uniform(0.9 * xyz_min[:2], 0.9 * xyz_max[:2])
        # xyz[2] = xyz_max[2] + 0.05
        #
        # obs, success, _, _ = env.step(np.array([*xyz, 0.0]))
        # print("SUCCESS", success)


if __name__ == "__main__":
    main()
