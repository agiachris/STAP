from os import path
import argparse
import json
import yaml
import copy
import tempfile
import itertools
import subprocess


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exec-config", "-e", type=str, required=True, help="Path to execution configs"
    )
    parser.add_argument(
        "--sweep-config",
        "-s",
        type=str,
        required=True,
        help="Path to parameter sweep config",
    )
    parser.add_argument(
        "--checkpoints",
        "-c",
        nargs="+",
        type=str,
        required=True,
        help="Path to model checkpoints",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to directory for saving results",
    )
    parser.add_argument(
        "--num-eps",
        "-n",
        type=int,
        default=100,
        help="Number of episodes to unit test across",
    )
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    # Load exec and sweep configs
    with open(args.exec_config, "r") as fs:
        exec_config = yaml.safe_load(fs)
    with open(args.sweep_config, "r") as fs:
        sweep_config = json.load(fs)
    assert exec_config["planner"] == sweep_config["planner"]
    fname = path.splitext(path.split(args.exec_config)[1])[0]
    fpath = path.join(args.path, fname)

    # Generate parameter assignment
    sweep_keys, sweep_values = zip(*list(sweep_config["planner_kwargs"].items()))
    assert all(isinstance(x, list) for x in sweep_values), "Sweep values must be lists"
    sweep_params = list(itertools.product(*sweep_values))
    assert all(len(x) == len(sweep_keys) for x in sweep_params)
    exec_configs = []
    for i, params in enumerate(sweep_params):
        exp_config = copy.deepcopy(exec_config)
        planner_kwargs = dict(zip(sweep_keys, params))
        exp_config["planner_kwargs"].update(planner_kwargs)

        with tempfile.NamedTemporaryFile(mode="w+") as ts:
            yaml.dump(exp_config, ts)
            cmd = ["python", "scripts/eval/eval_pybox2d.py"]
            cmd += ["--exec-config", ts.name, "--checkpoints"] + args.checkpoints
            cmd += [
                "--path",
                fpath + f"_{i}.json",
                "--num-eps",
                str(args.num_eps),
                "--device",
                args.device,
            ]
            subprocess.call(args=cmd)
