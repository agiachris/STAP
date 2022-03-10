import argparse
import os
from os import path
import yaml
import imageio
from copy import deepcopy
import numpy as np

import temporal_policies.algs.planners.pybox2d as pybox2d_planners
import temporal_policies.envs.pybox2d as pybox2d_envs
from temporal_policies.envs.pybox2d.visualization import PyBox2DVisualizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-config", type=str, required=True, help="Path to execution configs")
    parser.add_argument("--checkpoints", nargs="+", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--path", type=str, required=True, help="Path to save gifs and/or images")
    parser.add_argument("--gifs", action="store_true", help="Save GIF of the scene")
    parser.add_argument("--plot-2d", action="store_true", help="Plot 2D visualization of value estimates")
    parser.add_argument("--plot-3d", action="store_true", help="Plot 3D visualization of value estimates")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Save every n frames to the gif.")
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    with open(args.exec_config, "r") as fs: exec_config = yaml.safe_load(fs)
    env_cls = [vars(pybox2d_envs)[subtask["env"]] for subtask in exec_config["task"]]
    planner = vars(pybox2d_planners)[exec_config["planner"]](
        task=exec_config["task"],
        checkpoints=args.checkpoints,
        device=args.device,
        **exec_config["planner_kwargs"]
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
            curr_env = env(**config) if prev_env is None else env.load(prev_env, **config)
            
            # TODO: make generic
            if args.plot_3d and j < len(env_cls) - 1:
                
                # Current Q(s, a) over all actions (x, theta)
                temp_env = env_cls[j].clone(curr_env, **planner._get_config(j))
                state = temp_env._get_observation()
                curr_actions, curr_action_dims = temp_env._interp_actions(10, [0, 1])
                curr_q_vals = planner._q_function(j, state, curr_actions)

                # Store x, y, q
                x_curr = curr_actions[:, 0].copy()
                y_curr = curr_actions[:, 1].copy()
                z_curr = curr_q_vals.copy()

                # Simulate forward environments
                temp_envs = planner._clone_env(temp_env, j, num=curr_actions.shape[0])
                for temp_env, action in zip(temp_envs, curr_actions): planner._simulate_env(temp_env, action)
                temp_envs = [planner._load_env(temp_env, j+1) for temp_env in temp_envs]

                # Next Q(s, a)
                z_next = []
                for k, temp_env in enumerate(temp_envs):
                    state = temp_env._get_observation()
                    action = planner._policy(j+1, state)
                    z_next.append(planner._q_function(j+1, state, action).item())
                z_next = np.array(z_next)

                assert z_curr.shape == z_next.shape
                PyBox2DVisualizer.plot_xdim_theta_3d(
                    x_curr, 
                    y_curr, 
                    z_curr, 
                    z_next, 
                    [type(curr_env).__name__, type(temp_env).__name__],
                    path.join(args.path, f"example_{i}.png") 
                )
            break
            # for _ in range(curr_env._max_episode_steps):
            #     action = planner.plan(curr_env, j)
            #     obs, rew, done, info = curr_env.step(action)
            #     if done: break

            # if not info["success"]: break
            
        #     prev_env = curr_env
        
        # if args.gifs:
        #     frames = curr_env._frame_buffer + [curr_env._frame_buffer[-1]] * args.every_n_frames**2
        #     filepath = path.join(args.path, f"example_{i}.gif")
        #     imageio.mimsave(filepath, frames[::args.every_n_frames])
