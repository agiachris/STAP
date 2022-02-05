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
        curr_env = None

        for j in range(len(args.envs)):
            config = Config.load(args.configs[j])
            next_env = gym.make(args.envs[j] + "-v0", **config["env_kwargs"])
            if curr_env is None:
                _ = next_env.reset()
            else:
                next_env = type(next_env.unwrapped).load(curr_env, **config["env_kwargs"])

            draw_text(next_env.render(mode="rgb_array", width=args.width, height=args.height), \
                get_text(i, j, args.envs[j], step, reward)).show()

            for _ in range(next_env.unwrapped._max_episode_steps):
                action = next_env.action_space.sample()
                obs, rew, done, info = next_env.step(action)
                reward += rew
                step += 1
                if done: break
            if rew == 0: break
    
        draw_text(next_env.render(mode="rgb_array", width=args.width, height=args.height), \
                get_text(i, j, args.envs[j], step, reward)).show()
