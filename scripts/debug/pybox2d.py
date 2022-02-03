import numpy as np
from PIL import Image

from temporal_policies.envs.pybox2d import *


if __name__ == "__main__":

    # Example environment config
    env_kwargs = {
        "steps_per_action": 100,
        "env_params": {
            "spawned_block": {
                "class": "block",
                "type": "static",
                "shape_kwargs": {
                    "size": (0.5, 0.5),
                    "dx": -2,
                    "dy": 7
                },
            }
        },
        "rand_params": {}
    }

    # Unit test environment
    env = PlaceRight2D(**env_kwargs)
    obs = env.reset()
    obs, rew, done, info = env.step(np.random.uniform(-1, 1, 2))
    Image.fromarray(env.render(width=160, height=120), "RGB").show()
