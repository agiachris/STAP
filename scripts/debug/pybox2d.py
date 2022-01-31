import matplotlib.pyplot as plt

from temporal_policies.envs.pybox2d.pick2d import Pick2D


def plot(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    kwargs = {}
    env = Pick2D(**kwargs)
    _ = env.reset()

    plot(env.render())
    for i in range(60):
        _= env.step(None)
    plot(env.render())
    