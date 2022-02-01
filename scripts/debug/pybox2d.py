import numpy as np

from temporal_policies.envs.pybox2d import *
from temporal_policies.envs.pybox2d.utils import plot


if __name__ == "__main__":

    kwargs = {}
    env = PlaceRight2D(**kwargs)
    env._steps_per_action = 200
    for i in range(5):
        obs = env.reset()
        obs, rew, done, info = env.step(np.random.uniform(-1, 1, 2))
        plot(env.render())

    