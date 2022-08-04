#!/usr/bin/env python3

import argparse
import functools
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore

from temporal_policies import agents, envs, planners
from temporal_policies.utils import random, spaces, timing


def create_grid_policies(
    env: envs.SequentialEnv,
    policies: Sequence[agents.Agent],
    action_skeleton: Sequence[Tuple[int, Any]],
    grid_resolution: int,
) -> List[agents.Agent]:
    assert all(
        len(policy_env.action_space.shape) == 1 for policy_env in env.envs
    ), "Only vector actions supported"

    grid_policies: List[agents.Agent] = []
    for idx_policy, policy_args in action_skeleton[:-1]:
        policy_env = env.envs[idx_policy]
        action_space = policy_env.action_space
        actions = np.meshgrid(
            *np.linspace(action_space.low, action_space.high, grid_resolution).T
        )
        actions = np.stack(actions, axis=-1).reshape(-1, action_space.shape[0])

        grid_policies.append(
            agents.ConstantAgent(action=actions, policy=policies[idx_policy])
        )

    grid_policies.append(policies[-1])

    return grid_policies


def evaluate_critic_functions(
    planner: planners.Planner,
    action_skeleton: Sequence[Tuple[int, Any]],
    env: envs.SequentialEnv,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_policies = create_grid_policies(
        env, planner.policies, action_skeleton, grid_resolution
    )

    observation = env.get_observation(action_skeleton[0][0])
    with torch.no_grad():
        observation = torch.from_numpy(observation).to(planner.device)
        state = planner.dynamics.encode(observation, action_skeleton[0][0])
        state = state.repeat(grid_policies[0].actor.constant.shape[0], 1)
        states, actions, p_transitions = planner.dynamics.rollout(
            state=state,
            action_skeleton=action_skeleton,
            policies=grid_policies,
            time_index=True,
        )

        q_values = torch.zeros_like(p_transitions)
        for t, (idx_policy, policy_args) in enumerate(action_skeleton):
            policy_state = planner.dynamics.decode(
                states[:, t], idx_policy, policy_args
            )
            dim_action = torch.sum(~torch.isnan(actions[0, t])).cpu().item()
            action = actions[:, t, :dim_action]
            q_values[:, t] = planner.policies[idx_policy].critic.predict(
                policy_state, action
            )

    return q_values.cpu().numpy(), actions.cpu().numpy()


def plot_critic_functions(
    env: envs.pybox2d.Sequential2D,
    action_skeleton: Sequence[Tuple[int, Any]],
    actions: np.ndarray,
    p_success: np.ndarray,
    rewards: np.ndarray,
    grid_q_values: np.ndarray,
    grid_actions: np.ndarray,
    path: Union[str, pathlib.Path],
    title: Optional[str] = None,
) -> None:
    def tick_labels(value: float, pos: float, dim: int) -> str:
        x = np.array(env.envs[0].action_space.low)
        x[dim] = value
        x = spaces.transform(
            x,
            from_space=env.envs[0].action_space,
            to_space=env.envs[0].action_scale,
        )[dim]
        return f"{x:0.2f}"

    def plot_trisurf(
        ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, **kwargs
    ) -> None:
        ax.plot_trisurf(xs, ys, zs, cmap="plasma", linewidth=0, **kwargs)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("theta [rad]")

        action_space = env.envs[0].action_space
        ax.set_xlim(action_space.low[0], action_space.high[0])
        ax.set_ylim(action_space.low[1], action_space.high[1])
        ax.set_zlim(0, 1)

        xtick_labels = functools.partial(tick_labels, dim=0)
        ytick_labels = functools.partial(tick_labels, dim=1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(xtick_labels))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(ytick_labels))
        ax.set_xticks(np.linspace(action_space.low[0], action_space.high[0], 5))
        ax.set_yticks(np.linspace(action_space.low[1], action_space.high[1], 5))

    T = len(action_skeleton)
    fig, axes = plt.subplots(1, T + 1, subplot_kw={"projection": "3d"}, figsize=(16, 5))

    xs, ys = grid_actions[:, 0].T
    for t, (idx_policy, policy_args) in enumerate(action_skeleton):
        ax = axes[t]

        plot_trisurf(ax, xs, ys, grid_q_values[:, t])
        ax.set_title(f"{type(env.envs[idx_policy]).__name__} Q(s, a)")
        ax.set_zlabel("Q(s, a)")

    ax = axes[2]
    cmap = plt.get_cmap("tab10")
    idx_best = p_success.argmax()
    ax.scatter(
        *actions[idx_best, 0].T, p_success[idx_best], color=cmap(3), linewidth=10
    )
    ax.scatter(*actions[:, 0].T, p_success, color=cmap(2), marker=".", linewidth=0)
    plot_trisurf(ax, xs, ys, np.clip(grid_q_values, 0, 1).prod(axis=-1), alpha=0.5)

    ax.set_title(f"Predicted success: {p_success[idx_best]}\nGround truth: {rewards}")
    ax.set_zlabel("success prob")

    if title is not None:
        fig.suptitle(title)

    fig.savefig(path)
    plt.close(fig)


def scale_actions(
    actions: np.ndarray,
    env: envs.SequentialEnv,
    action_skeleton: Sequence[Tuple[int, Any]],
) -> np.ndarray:
    scaled_actions = actions.copy()
    for t, (idx_policy, policy_args) in enumerate(action_skeleton):
        policy_env = env.envs[idx_policy]
        action_dims = policy_env.action_space.shape[0]
        scaled_actions[..., t, :action_dims] = spaces.transform(
            actions[..., t, :action_dims],
            from_space=policy_env.action_space,
            to_space=policy_env.action_scale,
        )

    return scaled_actions


def evaluate_planners(
    config: Union[str, pathlib.Path],
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    grid_resolution: int,
    verbose: bool,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    timer = timing.Timer()

    env = envs.load(env_config)
    planner = planners.load(
        config=config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        scod_checkpoints=scod_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    path = pathlib.Path(path) / pathlib.Path(config).stem
    path.mkdir(parents=True, exist_ok=True)

    action_skeleton = [(0, None), (1, None)]

    for i in tqdm.tqdm(range(num_eval), f"Evaluate {path.name}", dynamic_ncols=True):
        if seed is not None:
            random.seed(i)

        observation = env.reset(action_skeleton[0][0])
        state = env.get_state()

        timer.tic("planner")
        actions, p_success, visited_actions, p_visited_success = planner.plan(
            observation, action_skeleton
        )
        t_planner = timer.toc("planner")

        rewards = planners.evaluate_plan(
            env,
            action_skeleton,
            state,
            actions,
            gif_path=path / f"exec_{i}.gif",
        )

        if verbose:
            print("success:", rewards.prod())
            print("predicted success:", p_success)
            print(actions)
            print("time:", t_planner)

        env.set_state(state)
        grid_q_values, grid_actions = evaluate_critic_functions(
            planner, action_skeleton, env, grid_resolution
        )
        plot_critic_functions(
            env=env,
            action_skeleton=action_skeleton,
            actions=visited_actions,
            p_success=p_visited_success,
            rewards=rewards,
            grid_q_values=grid_q_values,
            grid_actions=grid_actions,
            path=path / f"values_{i}.png",
            title=f"{pathlib.Path(config).stem}: {t_planner:0.2f}s",
        )

        with open(path / f"results_{i}.npz", "wb") as f:
            save_dict = {
                "args": {
                    "config": config,
                    "env_config": env_config,
                    "policy_checkpoints": policy_checkpoints,
                    "dynamics_checkpoint": dynamics_checkpoint,
                    "device": device,
                    "num_eval": num_eval,
                    "path": path,
                    "seed": seed,
                    "grid_resolution": grid_resolution,
                    "verbose": verbose,
                },
                "observation": observation,
                "state": state,
                "actions": actions,
                "scaled_actions": scale_actions(actions, env, action_skeleton),
                "p_success": p_success,
                "rewards": rewards,
                "grid_q_values": grid_q_values,
                "grid_actions": grid_actions,
                "visited_actions": visited_actions,
                "scaled_visited_actions": scale_actions(
                    visited_actions, env, action_skeleton
                ),
                "p_visited_success": p_visited_success,
                "t_planner": t_planner,
            }
            np.savez_compressed(f, **save_dict)


def main(args: argparse.Namespace) -> None:
    evaluate_planners(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "--planner-config", "--planner", "-c", help="Path to planner config"
    )
    parser.add_argument("--env-config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument(
        "--scod-checkpoints", "-s", nargs="+", help="SCOD checkpoints"
    )
    parser.add_argument("--dynamics-checkpoint", "-d", help="Dynamics checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=40,
        help="Resolution of critic function plot",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    args = parser.parse_args()

    main(args)