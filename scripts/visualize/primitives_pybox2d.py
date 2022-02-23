import argparse
import os
from os import path
import imageio

from temporal_policies.utils.config import Config
from temporal_policies.utils.trainer import load_from_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save gif")
    parser.add_argument("--checkpoints", nargs="+", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vis-random", action="store_true", help="Visualize a random policy baseline")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Save every n frames to the gif.")
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    models = [load_from_path(c, device=args.device, strict=True) for c in args.checkpoints]
    for model in models: model.eval_mode()
    configs = [Config.load(path.join(path.dirname(c), "config.yaml")) for c in args.checkpoints]

    # Outputs
    exps = ["rollout_policy"]
    if args.vis_random: exps.insert(0, "random_policy")
    paths = [path.join(args.path, exp) for exp in exps]
    for p in paths: 
        assert not path.exists(p), "Save path already exists"
        os.makedirs(p)

    # Simulate random and trained policies
    for e, exp in enumerate(exps):
        for i in range(args.num_eps):
            prev_env = None

            for j, model in enumerate(models):
                # Instantiate environment
                if prev_env is None:
                    curr_env = model.env.unwrapped
                    obs = curr_env.reset()
                else:
                    curr_env = type(model.env.unwrapped).load(prev_env, **configs[j]["env_kwargs"])
                    obs = curr_env._get_observation()
                
                # Simulate forward under policy
                for _ in range(curr_env._max_episode_steps):
                    if exp == "random_policy": action = curr_env.action_space.sample()
                    else: action = model.predict(obs)
                    obs, rew, done, info = curr_env.step(action)
                    if done: break
                if not info["success"]: break

                prev_env = curr_env

            frames = curr_env._frame_buffer + [curr_env._frame_buffer[-1]] * args.every_n_frames**2
            filepath = path.join(paths[e], f"example_{i}.gif")
            imageio.mimsave(filepath, frames[::args.every_n_frames])
