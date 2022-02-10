import argparse
import os
from os import path
import numpy as np
from PIL import Image

from temporal_policies.utils.config import Config
from temporal_policies.utils.trainer import load_from_path
from temporal_policies.envs.pybox2d.visualization import PyBox2DVisualizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save gif")
    parser.add_argument("--checkpoints", nargs="+", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of value estimates per image")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    models = [load_from_path(c, device=args.device, strict=True) for c in args.checkpoints]
    for model in models: model.eval_mode()
    configs = [Config.load(path.join(path.dirname(c), "config.yaml")) for c in args.checkpoints]

    # Outputs
    assert not path.exists(args.path), "Save path already exists"
    os.makedirs(args.path)

    # Simulate random and trained policies
    for i in range(args.num_eps):
        prev_env = None

        for j, model in enumerate(models):
            if prev_env is None:
                curr_env = model.env.unwrapped
                obs = curr_env.reset()
            else:
                curr_env = type(model.env.unwrapped).load(prev_env, **configs[j]["env_kwargs"])
                obs = curr_env._get_observation()
            
            if j + 1 < len(models):
                # Value estimates over agent x-dim component
                curr_qs, curr_outputs = curr_env.value_over_interp_actions(model, args.num_samples, [0])
                next_envs = [type(models[j+1].env.unwrapped).load(env, **configs[j+1]["env_kwargs"]) for env in curr_outputs["env"]]
                next_qs, next_outputs = zip(*[env.action_value(models[j+1]) for env in next_envs]) 

                visualizer = PyBox2DVisualizer(curr_env)
                visualizer.render_values_xdim(
                    x=[np.array(curr_outputs["action_dim"])],
                    y=[curr_qs, np.array(next_qs)],
                    labels=[type(curr_env).__name__, type(next_envs[0]).__name__],
                    colors=["blue", "orange"]
                )
                visualizer.save(path.join(args.path, f"sample_{i}_{j}_xdim.png"))


                # # Value estimates over theta component
                # curr_qs, curr_outputs = curr_env.value_over_interp_actions(model, args.num_samples, [1])
                # next_envs = [type(models[j+1].env.unwrapped).load(env, **configs[j+1]["env_kwargs"]) for env in curr_outputs["env"]]
                # next_qs, next_outputs = zip(*[env.action_value(models[j+1]) for env in next_envs]) 

                # plot_kwargs = {
                #     "x": np.array(curr_outputs["action_dim"]).squeeze(-1),
                #     "y": [curr_qs, np.array(next_qs)],
                #     "labels": [type(curr_env).__name__, type(next_envs[0]).__name__],
                #     "colors": ["tab:blue", "tab:red"],
                # }
                # image = Image.fromarray(curr_env.render(**plot_kwargs))
                # image.save(path.join(args.path, f"sample_{i}_{j}_theta"), "png")
            
            for _ in range(curr_env._max_episode_steps):
                action = model.predict(obs)
                obs, rew, done, info = curr_env.step(action, render=True)
                if done: break

            if rew == 0: break
            prev_env = curr_env
        