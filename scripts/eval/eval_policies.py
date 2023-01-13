#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import tqdm

from temporal_policies import agents, envs
from temporal_policies.utils import random, tensors


@tensors.numpy_wrap
def query_policy_actor(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.actor.predict(
        policy.encoder.encode(observation.to(policy.device), policy_args)
    )


@tensors.numpy_wrap
def query_policy_critic(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy critic."""
    action = policy.actor.predict(
        policy.encoder.encode(observation.to(policy.device), policy_args)
    )
    return policy.critic.predict(
        policy.encoder.encode(observation.to(policy.device), policy_args), action
    )


def observation_str(env: envs.Env, observation: np.ndarray) -> str:
    """Converts observations to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        return str(env.object_states())

    return str(observation)


def action_str(env: envs.Env, action: np.ndarray) -> str:
    """Converts actions to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        primitive = env.get_primitive()
        assert isinstance(primitive, envs.pybullet.table.primitives.Primitive)
        return str(primitive.Action(action))

    return str(action)


def evaluate_episode(
    policy: agents.RLAgent,
    env: envs.Env,
    seed: Optional[int] = None,
    verbose: bool = False,
    debug: bool = False,
    record: bool = False,
) -> List[float]:
    """Evaluates the policy on one episode."""
    observation, reset_info = env.reset(seed=seed)
    if verbose:
        print("primitive:", env.get_primitive())
        print("reset_info:", reset_info)

    if record:
        env.record_start()

    t_pos = 0
    t_neg = 0
    f_pos = 0
    f_neg = 0
    rewards = []
    obs_diffs = []
    done = False
    while not done:
        action = query_policy_actor(policy, observation, reset_info["policy_args"])
        q = query_policy_critic(policy, observation, reset_info["policy_args"])
        if debug:
            input("step?")

        new_observation, reward, terminated, truncated, step_info = env.step(action)
        if q > 0.5:
            if reward > 0:
                print(f"True +ve q: {q}")
                t_pos += 1
            else:
                print(f"False +ve: {q}")
                f_pos += 1
        if q < 0.5:
            if reward > 0:
                print(f"False -ve: {q}")
                f_neg += 1
            else:
                print(f"True -ve: {q}")
                t_neg += 1

        if verbose:
            print("step_info:", step_info)
            print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
            # print magnitude of difference in observations
            print(
                "observation diff w/o hook:",
                np.linalg.norm(observation - new_observation),
            )
            obs_diffs.append(np.linalg.norm(observation - new_observation))
        observation = new_observation
        rewards.append(reward)
        done = terminated or truncated

    if record:
        env.record_stop()

    if debug:
        input("finish?")

    info = {
        "obs_diff": obs_diffs,
        "t_pos": t_pos,
        "t_neg": t_neg,
        "f_pos": f_pos,
        "f_neg": f_neg,
    }
    return rewards, info


def evaluate_episodes(
    env: envs.Env,
    policy: agents.RLAgent,
    num_episodes: int,
    path: pathlib.Path,
    verbose: bool,
) -> None:
    """Evaluates policy for the given number of episodes."""
    num_successes = 0
    pbar = tqdm.tqdm(
        range(num_episodes),
        desc=f"Evaluate {env.name}",
        dynamic_ncols=True,
    )
    rewards_lst = []
    obs_diffs = []
    t_pos, t_neg, f_pos, f_neg = 0, 0, 0, 0
    for i in pbar:
        # Evaluate episode.
        rewards, info = evaluate_episode(
            policy, env, verbose=verbose, debug=False, record=True
        )
        obs_diff = info["obs_diff"]
        t_pos += info["t_pos"]
        t_neg += info["t_neg"]
        f_pos += info["f_pos"]
        f_neg += info["f_neg"]
        rewards_lst.append(rewards)
        obs_diffs.append(obs_diff)
        success = sum(rewards) > 0.0
        num_successes += success
        pbar.set_postfix(
            {"rewards": rewards, "successes": f"{num_successes} / {num_episodes}"}
        )

        # Save recording.
        suffix = "" if success else "_fail"
        env.record_save(path / env.name / f"eval_{i}{suffix}.gif", reset=True)

        if isinstance(env, envs.pybullet.TableEnv):
            # Save reset seed.
            with open(path / env.name / f"results_{i}.npz", "wb") as f:
                save_dict = {
                    "seed": env._seed,
                }
                np.savez_compressed(f, **save_dict)  # type: ignore
    if verbose:
        rewards_lst = np.array(rewards_lst)
        obs_diffs = np.array(obs_diffs)
        print(f"t_pos: {t_pos}, t_neg: {t_neg}, f_pos: {f_pos}, f_neg: {f_neg}")
        print(f"obs_diffs: {obs_diffs}")
        print(f"obs_diffs avg: {np.mean(obs_diffs)}")
        print(f"obs_diffs std: {np.std(obs_diffs)}")
        # stats for successes
        success_obs_diffs = obs_diffs[rewards_lst > 0]
        print(f"success_obs_diffs: {success_obs_diffs}")
        print(f"success_obs_diffs avg: {np.mean(success_obs_diffs)}")
        print(f"success_obs_diffs std: {np.std(success_obs_diffs)}")
        # stats for failures
        fail_obs_diffs = obs_diffs[rewards_lst <= 0]
        print(f"fail_obs_diffs: {fail_obs_diffs}")
        print(f"fail_obs_diffs avg: {np.mean(fail_obs_diffs)}")
        print(f"fail_obs_diffs std: {np.std(fail_obs_diffs)}")


def evaluate_policy(
    checkpoint: Union[str, pathlib.Path],
    env_config: Optional[Union[str, pathlib.Path]] = None,
    debug_results: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    num_episodes: int = 100,
    seed: Optional[int] = 0,
    gui: Optional[bool] = None,
    verbose: bool = True,
    device: str = "auto",
) -> None:
    """Evaluates the policy either by loading an episode from `debug_results` or
    generating `num_eval_episodes` episodes.
    """
    if path is None and debug_results is None:
        raise ValueError("Either path or load_results must be specified")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load env.
    env_kwargs: Dict[str, Any] = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    if env_config is None:
        # Try to load eval env.
        env_config = pathlib.Path(checkpoint).parent / "eval/env_config.yaml"
    try:
        env = envs.load(env_config, **env_kwargs)
    except FileNotFoundError:
        # Default to train env.
        env = None

    # Load policy.
    policy = agents.load(
        checkpoint=checkpoint, env=env, env_kwargs=env_kwargs, device=device
    )
    assert isinstance(policy, agents.RLAgent)
    policy.eval_mode()

    if env is None:
        env = policy.env

    if debug_results is not None:
        # Load reset seed.
        with open(debug_results, "rb") as f:
            seed = int(np.load(f, allow_pickle=True)["seed"])
        evaluate_episode(
            policy, env, seed=seed, verbose=verbose, debug=True, record=False
        )
    elif path is not None:
        # Evaluate episodes.
        evaluate_episodes(env, policy, num_episodes, pathlib.Path(path), verbose)


def main(args: argparse.Namespace) -> None:
    evaluate_policy(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--debug-results", type=str, help="Path to results_i.npz file.")
    parser.add_argument("--path", help="Path for output plots")
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--device", default="auto", help="Torch device")
    args = parser.parse_args()

    main(args)
