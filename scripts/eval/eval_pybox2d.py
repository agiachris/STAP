import time
import argparse
from os import path
import numpy as np
import yaml

from temporal_policies.utils.config import Config
import temporal_policies.algs.planners.pybox2d as pybox2d_planners
import temporal_policies.envs.pybox2d as pybox2d_envs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-config", type=str, required=True, help="Path to execution configs")
    parser.add_argument("--checkpoints", nargs="+", type=str, required=True, help="Path to model checkpoints")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    with open(args.exec_config, "r") as fs: exec_config = yaml.safe_load(fs)
    env_cls = [vars(pybox2d_envs)[subtask["env"]] for subtask in exec_config["task"]]
    configs = [Config.load(path.join(path.dirname(c), "config.yaml")) for c in args.checkpoints]
    planner = vars(pybox2d_planners)[exec_config["planner"]](
        task=exec_config["task"],
        checkpoints=args.checkpoints,
        configs=configs,
        device=args.device,
        **exec_config["planner_kwargs"]
    )
    
    # Evaluate
    ep_rewards = np.zeros(args.num_eps)
    micro_steps = np.zeros(args.num_eps)
    macro_steps = np.zeros(args.num_eps)
    time_per_primitive = np.zeros(args.num_eps)

    for i in range(args.num_eps):
        step = 0
        reward = 0
        prev_env = None
        ep_time = 0

        for j, (env, config) in enumerate(zip(env_cls, configs)):
            curr_env = env(**config["env_kwargs"]) if prev_env is None \
                else env.load(prev_env, **config["env_kwargs"])
            
            st = time.time()
            for _ in range(curr_env._max_episode_steps):
                action = planner.plan(curr_env, j)
                obs, rew, done, info = curr_env.step(action)
                reward += rew
                step += 1
                if done: break

            ep_time += time.time() - st
            if not info["success"]: break
            
            prev_env = curr_env

        ep_rewards[i] = reward
        micro_steps[i] = step
        macro_steps[i] = j
        time_per_primitive[i] = ep_time / (j + 1)
    
    reward_min = np.amin(ep_rewards)
    reward_max = np.amax(ep_rewards)
    # Compute stats
    print(f"Results for {args.exec_config} policy over {i+1} episodes:")
    print(f"\tRewards: mean {ep_rewards.mean():.2f} std {ep_rewards.std():.2f}")
    print(f"\t         min {reward_min} percent {(ep_rewards == reward_min).sum() / (i+1)}")
    print(f"\t         max {reward_max} percent {(ep_rewards == reward_max).sum() / (i+1)}")
    print(f"\tSec / Primitive: mean {time_per_primitive.mean():.2f} std {time_per_primitive.std():.2f}")
    print(f"\tPrimitives: mean {macro_steps.mean():.2f} std {macro_steps.std():.2f}")
    print(f"\tSteps: mean {micro_steps.mean():.2f} std {micro_steps.std():.2f}\n")
