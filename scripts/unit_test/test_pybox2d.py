import argparse
import numpy as np
import gym
from PIL import Image

from temporal_policies.utils.config import Config
from temporal_policies.envs.pybox2d.visualization import draw_caption


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    args = parser.parse_args()

    for i in range(args.num_eps):
        step = 0
        reward = 0
        prev_env = None

        for j in range(len(args.configs)):
            config = Config.load(args.configs[j])
            env_name = config["env"].split("-")[0]
            curr_env = gym.make(env_name + "-v0", **config["env_kwargs"]).unwrapped
            if prev_env is None:
                obs = curr_env.reset()
            else:
                curr_env = type(curr_env).load(prev_env, **config["env_kwargs"])
                obs = curr_env._get_observation()
            curr_env_clone = type(curr_env).clone(curr_env, **config["env_kwargs"])

            Image.fromarray(curr_env.render()).show()
            draw_caption(curr_env_clone.render(), "Cloned environment", loc="center").show()

            for _ in range(curr_env.unwrapped._max_episode_steps):
                action = curr_env.action_space.sample()
                obs, rew, done, info = curr_env.step(action)
                obs_c, rew_c, done_c, info_c = curr_env_clone.step(action)
                assert np.all(obs == obs_c) and rew == rew_c and done == done_c, "Original and cloned environments diverged"
                reward += rew
                step += 1
                if done: break
            if rew == 0: break
            prev_env = curr_env

        Image.fromarray(curr_env.render()).show()
        draw_caption(curr_env_clone.render(), "Cloned environment", loc="center").show()
