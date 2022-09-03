import os
from os import path
import time
import argparse
import numpy as np
import yaml
import json
from copy import deepcopy
import pprint

import temporal_policies.algs.planners.pybox2d as pybox2d_planners
import temporal_policies.envs.pybox2d as pybox2d_envs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exec-config", "-e", type=str, required=True, help="Path to execution configs"
    )
    parser.add_argument(
        "--policy_checkpoints",
        "-c",
        nargs="+",
        type=str,
        required=True,
        help="Path to model checkpoints",
    )
    parser.add_argument(
        "--dynamics_checkpoint",
        "-d",
        type=str,
        required=False,
        help="Path to dynamics model checkpoints",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to save json files of results",
    )
    parser.add_argument(
        "--num-eps",
        "-n",
        type=int,
        default=1,
        help="Number of episodes to unit test across",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    with open(args.exec_config, "r") as fs:
        exec_config = yaml.safe_load(fs)
    env_cls = [vars(pybox2d_envs)[subtask["env"]] for subtask in exec_config["task"]]
    planner = vars(pybox2d_planners)[exec_config["planner"]](
        task=exec_config["task"],
        policy_checkpoints=args.policy_checkpoints,
        dynamics_checkpoint=args.dynamics_checkpoint,
        device=args.device,
        **exec_config["planner_kwargs"],
    )
    if not path.splitext(args.path)[-1] == ".json":
        args.path += ".json"
    assert not path.exists(args.path), "Save path already exists"
    fdir = path.dirname(args.path)
    if not path.exists(fdir):
        os.makedirs(fdir)

    # Evaluate
    ep_rewards = np.zeros(args.num_eps)
    micro_steps = np.zeros(args.num_eps)
    macro_steps = np.zeros(args.num_eps)
    time_per_primitive = np.zeros(args.num_eps)

    for i in range(args.num_eps):
        step = 0
        reward = 0
        ep_time = 0.0
        prev_env = None
        for j, env in enumerate(env_cls):
            config = deepcopy(planner._get_config(j))
            curr_env = (
                env(**config) if prev_env is None else env.load(prev_env, **config)
            )

            st = time.time()
            for _ in range(curr_env._max_episode_steps):
                action = planner.plan(j, curr_env)
                obs, rew, terminated, truncated, info = curr_env.step(action)
                reward += rew
                step += 1
                if terminated or truncated:
                    break

            ep_time += time.time() - st
            if not info["success"]:
                break

            prev_env = curr_env

        ep_rewards[i] = reward
        micro_steps[i] = step
        macro_steps[i] = j + 1
        time_per_primitive[i] = ep_time / (j + 1)

    # Log results
    results = {"settings": planner.planner_settings}
    results["return_mean"] = ep_rewards.mean()
    results["return_std"] = ep_rewards.std()
    results["return_min"] = ep_rewards.min()
    results["return_max"] = ep_rewards.max()
    results["return_min_percentage"] = (ep_rewards == ep_rewards.min()).sum() / (i + 1)
    results["return_max_percentage"] = (ep_rewards == ep_rewards.max()).sum() / (i + 1)
    results["frequency_mean"] = (1 / time_per_primitive).mean()
    results["frequency_std"] = (1 / time_per_primitive).std()
    results["primitives_mean"] = macro_steps.mean()
    results["primitives_std"] = macro_steps.std()
    results["steps_mean"] = micro_steps.mean()
    results["steps_std"] = micro_steps.std()
    print(f"Results for {path.split(args.exec_config)[1]} over {i+1} runs:")
    pprint.pprint(results, indent=4)
    with open(args.path, "w") as fs:
        json.dump(results, fs)
