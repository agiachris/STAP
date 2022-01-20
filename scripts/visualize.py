import argparse
import imageio
from research.utils.trainer import load_from_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to save the gif")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--num-ep", type=int, default=1, help="Number of episodes")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Save every n frames to the gif.")
    parser.add_argument("--width", type=int, default=160, help="Width of image")
    parser.add_argument("--height", type=int, default=120, help="Height of image")
    args = parser.parse_args()

    model = load_from_path(args.checkpoint, device=args.device, strict=True)
    env = model.env

    frames = []
    for ep in range(args.num_ep):
        obs = env.reset()
        done = False
        frames.append(env.render(mode='rgb_array', width=args.width, height=args.height))
        while not done:
            action = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            frames.append(env.render(mode='rgb_array', width=args.width, height=args.height))

    # Cut the frames 
    print("Saving a gif of", len(frames), "Frames")
    imageio.mimsave(args.path, frames[::args.every_n_frames])
    
