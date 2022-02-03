import numpy as np
from PIL import Image

from temporal_policies.envs.pybox2d import PlaceRight2D


if __name__ == "__main__":

    # Example environment config
    env_kwargs = {
        "max_episode_steps": 1,
        "steps_per_action": 100,
        "env_params": {
            "spawned_block": {
                "class": "block",
                "type": "static",
                "shape_kwargs": {
                    "size": (0.5, 0.5),
                    "dx": -4,
                    "dy": 7
                }
            }
        },
        "rand_params": {
            "obstacle": {
                "shape_kwargs": {
                    "dx": [0.0, 5.0]
                }
            }
        }
    }

    # Unit test environment
    env = PlaceRight2D(**env_kwargs)
    obs = env.reset()
    Image.fromarray(env.render()).show()
    obs, rew, done, info = env.step(np.random.uniform(-1, 1, 2))
    Image.fromarray(env.render()).show()

    # Load environment state from pre-existing
    env = PlaceRight2D.load(env, **env_kwargs)
    Image.fromarray(env.render()).show()
