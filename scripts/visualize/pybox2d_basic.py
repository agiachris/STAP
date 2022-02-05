from os import path
import argparse
import numpy as np
import imageio

from temporal_policies.utils.trainer import load_from_path
from temporal_policies.envs.pybox2d import (PlaceRight2D, PushLeft2D)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save the gif")
    parser.add_argument("--checkpoint_1", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--checkpoint_2", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--num-ep", type=int, default=100, help="Number of episodes")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Save every n frames to the gif.")
    parser.add_argument("--width", type=int, default=160, help="Width of image")
    parser.add_argument("--height", type=int, default=120, help="Height of image")
    args = parser.parse_args()

    random_rewards = np.zeros(args.num_ep)
    policy_rewards = np.zeros(args.num_ep)
    for i in range(args.num_ep):

        ## Test Random
        # Initialize PlaceRight2D environment
        frames = []
        env_kwargs = {
            "max_episode_steps": 1,
            "steps_per_action": 100,
            "rand_params": {
                "obstacle": {"shape_kwargs": {"dx": [0.0, 5.0]}}
            }
        }
        env = PlaceRight2D(**env_kwargs)
        obs = env.reset()
        obs, rew, done, info = env.step(np.random.uniform(-1, 1, 2), render=True)
        random_rewards[i] += rew
        frames.extend(env._render_buffer)
        # Load environment state into PushLeft2D
        env_kwargs = {
            "max_episode_steps": 50,
            "steps_per_action": 5,
        }
        env = PushLeft2D.load(env, **env_kwargs)
        for s in range(env_kwargs["max_episode_steps"]):
            obs, rew, done, info = env.step(np.random.uniform(-1, 1, 1), render=True)
            if done:
                break
        random_rewards[i] += rew
        frames.extend(env._render_buffer)
        
        filepath = path.join(args.path, "random2", "sample_{}.gif".format(i))
        imageio.mimsave(filepath, frames[::args.every_n_frames])

        ## Test Policy
        frames = []
        model_1 = load_from_path(args.checkpoint_1, device=args.device, strict=True)
        env_kwargs = {
            "max_episode_steps": 1,
            "steps_per_action": 100,
            "rand_params": {
                "obstacle": {"shape_kwargs": {"dx": [0.0, 5.0]}}
            }
        }
        env = PlaceRight2D(**env_kwargs)
        obs = env.reset()
        action = model_1.predict(obs)
        obs, rew, done, info = env.step(action, render=True)
        policy_rewards[i] += rew
        frames.extend(env._render_buffer)

        model_2 = load_from_path(args.checkpoint_2, device=args.device, strict=True)
        env_kwargs = {
            "max_episode_steps": 50,
            "steps_per_action": 5,
        }
        env = PushLeft2D.load(env, **env_kwargs)
        obs = env._get_observation()
        for s in range(env_kwargs["max_episode_steps"]):
            action = model_2.predict(obs)
            obs, rew, done, info = env.step(action, render=True)
            if done:
                break
        policy_rewards[i] += rew
        frames.extend(env._render_buffer)

        policy = path.splitext(path.dirname(args.checkpoint_1))[0].split("_")[-1].lower()
        filepath = path.join(args.path, policy, "sample_{}.gif".format(i))
        imageio.mimsave(filepath, frames[::args.every_n_frames])

    print("Random Policy Rewards: mean {} std {}".format(random_rewards.mean(), np.std(random_rewards)))
    print("Primitive Policy Rewards: mean {} std {}".format(policy_rewards.mean(), np.std(policy_rewards)))
