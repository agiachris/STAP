import argparse
import os
from os import path
import yaml
import imageio
from copy import deepcopy

import temporal_policies.algs.planners.pybox2d as pybox2d_planners
import temporal_policies.envs.pybox2d as pybox2d_envs
from temporal_policies.envs.pybox2d.visualization import Box2DVisualizer, plot_toy_demo


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exec-config", type=str, required=True, help="Path to execution configs"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory for saving gifs and/or images",
    )
    parser.add_argument(
        "--num-eps", type=int, default=1, help="Number of episodes to unit test across"
    )
    parser.add_argument("--device", "-d", type=str, default="auto")
    # Gif Visualization Arguments
    parser.add_argument("--gifs", action="store_true", help="Save GIF of the scene")
    parser.add_argument(
        "--every-n-frames", type=int, default=10, help="Save every n frames to the gif."
    )
    # 2D-3D Visualization Arguments
    parser.add_argument(
        "--plot-2d",
        action="store_true",
        help="Plot 2D visualization of value estimates",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Plot 3D visualization of value estimates",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=10,
        help="Discretization along state / action space components",
    )
    parser.add_argument(
        "--plot-unbiased",
        type=bool,
        default=True,
        help="Evaluate Q(s, a) action components under learned policy",
    )
    args = parser.parse_args()

    # Setup
    with open(args.exec_config, "r") as fs:
        exec_config = yaml.safe_load(fs)
    env_cls = [vars(pybox2d_envs)[subtask["env"]] for subtask in exec_config["task"]]
    planner = vars(pybox2d_planners)[exec_config["planner"]](
        task=exec_config["task"],
        checkpoints=args.checkpoints,
        device=args.device,
        **exec_config["planner_kwargs"],
    )
    assert not path.exists(args.path), "Save path already exists"
    os.makedirs(args.path)

    # Evaluate
    for i in range(args.num_eps):
        print(f"Episode {i+1} / {args.num_eps}")

        prev_env = None
        for j, env in enumerate(env_cls):
            config = deepcopy(planner._get_config(j))
            config["buffer_frames"] = True
            curr_env = (
                env(**config) if prev_env is None else env.load(prev_env, **config)
            )

            # Note: 2D and 3D visualization only available for toy PlaceRight2D + PushLeft2D<control> task
            if (args.plot_3d or args.plot_2d) and j == 0:
                temp_env = env.clone(curr_env, **config)
                visualizer = Box2DVisualizer(temp_env)
                plot_toy_demo(
                    episode=i,
                    visualizer=visualizer,
                    env=temp_env,
                    planner=planner,
                    output_path=args.path,
                    samples=args.plot_samples,
                    ensure_unbiased=args.plot_unbiased,
                    plot_2d=args.plot_2d,
                    plot_3d=args.plot_3d,
                )

            for _ in range(curr_env._max_episode_steps):
                action = planner.plan(j, curr_env)
                obs, rew, terminated, truncated, info = curr_env.step(action)
                if terminated or truncated:
                    break

            if not info["success"]:
                break

            prev_env = curr_env

        if args.gifs:
            frames = (
                curr_env._frame_buffer
                + [curr_env._frame_buffer[-1]] * args.every_n_frames**2
            )
            filepath = path.join(args.path, f"example_{i}.gif")
            imageio.mimsave(filepath, frames[:: args.every_n_frames])
