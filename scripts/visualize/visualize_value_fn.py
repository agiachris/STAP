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
    configs = [Config.load(path.join(path.dirname(c), "config.yaml")) for c in args.checkpoints]

    # Outputs
    exps = ["learned_policy"]
    if args.eval_random: exps.insert(0, "random_policy")
    paths = [path.join(args.path, exp) for exp in exps]
    for p in paths: 
        assert not path.exists(p), "Save path already exists"
        os.makedirs(p)

    # Simulate random and trained policies
    for e, exp in enumerate(exps):
        ep_rewards = np.zeros(args.num_eps)
        micro_steps = np.zeros(args.num_eps)
        macro_steps = np.zeros(args.num_eps)

        for i in range(args.num_eps):
            step = 0
            reward = 0
            frames = []
            prev_env = None

            for j, model in enumerate(models):
                if prev_env is None:
                    curr_env = model.env.unwrapped
                    obs = curr_env.reset()
                else:
                    curr_env = type(model.env.unwrapped).load(prev_env, **configs[j]["env_kwargs"])
                    obs = curr_env._get_observation()

                # PlaceRight2D, PushLeft2D
                # 1. PlaceRight2D, Q(s, a), s constant, a ranges | a = (x, theta) 
                # 2. PushLeft2D, Q(s, a), a constant, s ranges | s = (x, theta)

                for _ in range(curr_env._max_episode_steps):
                    if exp == "random_policy": action = curr_env.action_space.sample()
                    else: action = model.predict(obs)
                    obs, rew, done, info = curr_env.step(action, render=True)
                    reward += rew
                    step += 1
                    if done: break

                frames.extend(curr_env._render_buffer)
                if rew == 0: break
                prev_env = curr_env

            ep_rewards[i] = reward
            micro_steps[i] = step
            macro_steps[i] = j
            
            filepath = path.join(paths[e], f"sample_{i}.gif")
            imageio.mimsave(filepath, frames[::args.every_n_frames])

        # Compute stats
        print(f"Results for {exp} policy over {i+1} episodes")
        print(f"\tRewards: mean {ep_rewards.mean():.3f} std {ep_rewards.std():.3f}")
        print(f"\tPrimitives: mean {macro_steps.mean():.3f} std {macro_steps.std():.3f}")
        print(f"\tSteps: mean {micro_steps.mean():.3f} std {micro_steps.std():.3f}\n")
