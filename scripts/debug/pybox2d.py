import matplotlib.pyplot as plt
from temporal_policies.envs.pybox2d.pick2d import Pick2D


if __name__ == "__main__":

    kwargs = {}
    env = Pick2D(**kwargs)
    _ = env.reset()
    plt.imshow(env.render())
    plt.show()
    body = env._env_objects["item"]["bodies"]["block"]
    for i in range(60):
        _= env.step(None)
    plt.imshow(env.render())
    plt.show()
