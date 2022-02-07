import argparse
from PIL import (Image, ImageDraw)
import numpy as np
import gym

from temporal_policies import envs
from temporal_policies.utils.config import Config


def draw_text(image, text):
    """Draw text on image.
    args:
        image: RGB image as np.array HxWx3
        text: str text
    returns:
        image: PIL.Image
    """
    image = Image.fromarray(image)
    d = ImageDraw.Draw(image)
    d.text((10, 0), text, (0, 0, 0))
    return image


def get_text(ep, macro_step, env, micro_step, reward):
    return f"Ep: {ep} | Env: {macro_step}:{env} | Step: {micro_step} | Reward: {reward}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", type=str, required=True, help="Gym environment id")
    parser.add_argument("--configs", nargs="+", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--num-eps", type=int, default=1, help="Number of episodes to unit test across")
    parser.add_argument("--width", type=int, default=320, help="Pixel width of environment")
    parser.add_argument("--height", type=int, default=240, help="Pixel height of environment")
    args = parser.parse_args()

    for i in range(args.num_eps):
        step = 0
        reward = 0
        prev_env = None

        for j in range(len(args.envs)):
            config = Config.load(args.configs[j])
            curr_env = gym.make(args.envs[j] + "-v0", **config["env_kwargs"]).unwrapped
            if prev_env is None:
                obs = curr_env.reset()
            else:
                curr_env = type(curr_env).load(prev_env, **config["env_kwargs"])
                obs = curr_env._get_observation()

            draw_text(curr_env.render(mode="rgb_array", width=args.width, height=args.height), \
                get_text(i, j, args.envs[j], step, reward)).show()

            for _ in range(curr_env.unwrapped._max_episode_steps):
                action = curr_env.action_space.sample()
                obs, rew, done, info = curr_env.step(action)
                reward += rew
                step += 1
                if done: break
            if rew == 0: break
            prev_env = curr_env

        draw_text(curr_env.render(mode="rgb_array", width=args.width, height=args.height), \
                get_text(i, j, args.envs[j], step, reward)).show()
