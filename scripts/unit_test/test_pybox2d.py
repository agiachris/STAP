import os
import argparse
import numpy as np
import gym
import imageio

from temporal_policies.utils.config import Config
from temporal_policies.envs.pybox2d.visualization import draw_caption


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save gif")
    parser.add_argument("--configs", nargs="+", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Save every n frames to the gif.")
    args = parser.parse_args()

    assert not os.path.exists(args.path), "Save path already exists"
    os.makedirs(args.path)

    for i in range(args.num_eps):
        print(f"Episode {i+1} / {args.num_eps}")
        prev_env = None

        for j in range(len(args.configs)):
            # Load config
            config = Config.load(args.configs[j])
            env_kwargs = config["env_kwargs"]
            env_kwargs["buffer_frames"] = True
            env_kwargs["physics_steps_buffer"] = 10
            
            # Instantiate environment
            curr_env = gym.make(config["env"], **env_kwargs).unwrapped
            if prev_env is None:
                obs = curr_env.reset()
            else:
                curr_env = type(curr_env).load(prev_env, **env_kwargs)
                obs = curr_env._get_observation()
            curr_env_clone = type(curr_env).clone(curr_env, **env_kwargs)

            # Simulate forward randomly
            for _ in range(curr_env.unwrapped._max_episode_steps):
                action = curr_env.action_space.sample()
                obs, rew, done, info = curr_env.step(action)
                if done: break
            if not info["success"]: break

            prev_env = curr_env

        # Save frames
        filepath = os.path.join(args.path, f"example_{i}.gif")
        frames = curr_env._frame_buffer + [curr_env._frame_buffer[-1]] * args.every_n_frames**2
        imageio.mimsave(filepath, frames[::args.every_n_frames])
