import argparse
import os
from os import path
import imageio
import numpy as np

from temporal_policies.utils.config import Config
from temporal_policies.utils.trainer import load_from_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save gif")
    parser.add_argument("--checkpoints", nargs="+", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Save every n frames to the gif.")
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    models = [load_from_path(c, device=args.device, strict=True) for c in args.checkpoints]
    configs = [Config.load(path.join(path.dirname(c), "config.yaml") for c in args.checkpoints)]

    # Outputs
    model_name = path.splitext(path.dirname(args.checkpoints[0]))[0].split("_")[-1].lower()
    exps = ["random", model_name]
    prefix = ""
    for c in args.checkpoints: prefix += path.splitext(path.dirname(c)) + "_"
    prefix = prefix[:-1]
    paths = [path.join(args.path, prefix, exp) for exp in exps]
    for p in paths: 
        if not path.exists(p): os.makedirs(p)

    # Simulate random and trained policies
    for e, exp in enumerate(exps):
        ep_rewards = np.zeros(args.num_eps)
        micro_steps = np.zeros(args.num_eps)
        macro_steps = np.zeros(args.num_eps)

        for i in range(args.num_eps):
            step = 0
            reward = 0
            frames = []
            curr_env = None

            for j, model in enumerate(models):
                next_env = model.env
                if curr_env is None:
                    obs = next_env.reset()
                else:
                    next_env = type(next_env.unwrapped).load(curr_env, **configs[j]["env_kwargs"])
                    obs = next_env._get_observation()
                
                predict = next_env.action_space.sample if exp == "random" else model.predict
                for _ in range(next_env.unwrapped._max_episode_steps):
                    action = predict(obs)
                    obs, rew, done, info = next_env.step(action, render=True)
                    reward += rew
                    step += 1
                    if done: break

                frames.extend(next_env._render_buffer)
                if rew == 0: break

            ep_rewards[i] = reward
            micro_steps[i] = step
            macro_steps[i] = j
            
            filepath = path.join(paths[e], f"sample_{i}.gif")
            imageio.mimsave(filepath, frames[::args.every_n_frames])

        # Compute stats
        print(f"Results for {exp} policy over {i} episodes")
        print(f"\tRewards: mean {ep_rewards.mean():.3f} std {ep_rewards.std():.3f}")
        print(f"\tPrimitives: mean {macro_steps.mean():.3f} std {macro_steps.std():.3f}")
        print(f"\tSteps: mean {micro_steps.mean():.3f} std {micro_steps.std():.3f}\n")
