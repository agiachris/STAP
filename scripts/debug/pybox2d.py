from temporal_policies.envs.pybox2d.pick2d import Pick2D


if __name__ == "__main__":

    kwargs = {}
    env = Pick2D(**kwargs)
    _ = env.reset()
    body = env._env_objects["item"]["bodies"]["block"]
    for i in range(60):
        _, _, _, _ = env.step(None)
        print(body.position)
