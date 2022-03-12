import argparse
import pathlib

from temporal_policies.utils.trainer import Config, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    config = Config.load(args.config)
    policy = train(config, args.path, device=args.device)

    # Save replay buffers.
    replay_buffers_path = pathlib.Path(args.path) / "replay_buffers"
    policy.dataset.save(replay_buffers_path / "train")
    policy.eval_dataset.save(replay_buffers_path / "eval")
