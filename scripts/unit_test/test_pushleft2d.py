import numpy as np
from PIL import Image

from temporal_policies.envs.pybox2d import PushLeft2D


if __name__ == "__main__":

    # Example environment config
    env_kwargs = {
        "max_episode_steps": 20,
        "steps_per_action": 5,
        "env_params": {
            "spawned_block": {
                "class": "block",
                "type": "static",
                "shape_kwargs": {
                    "size": [0.5, 0.5],
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
            },
            "item": {
                "shape_kwargs": {
                    "dx": [0.0, 5.0]
                }
            }
        }
    }

    # Unit test environment
    env = PushLeft2D(**env_kwargs)
    obs = env.reset()
    for s in range(env_kwargs["max_episode_steps"]):
        obs, rew, done, info = env.step(np.random.uniform(-1, 1, 1))
        Image.fromarray(env.render()).show()
        if done:
            break
    # Load environment state from pre-existing
    env = PushLeft2D.load(env, **env_kwargs)
    Image.fromarray(env.render()).show()
