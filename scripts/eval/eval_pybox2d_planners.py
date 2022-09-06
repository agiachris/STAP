import functools
import pathlib
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import torch

from temporal_policies import agents, envs, planners
from temporal_policies.utils import spaces


def create_grid_policies(
    env: envs.Env,
    policies: Sequence[agents.Agent],
    action_skeleton: Sequence[envs.Primitive],
    grid_resolution: int,
) -> List[agents.Agent]:
    assert all(
        len(primitive.action_space.shape) == 1 for primitive in action_skeleton
    ), "Only vector actions supported"

    grid_policies: List[agents.Agent] = []
    for primitive in action_skeleton[:-1]:
        action_space = primitive.action_space
        action_mesh = np.meshgrid(
            *np.linspace(action_space.low, action_space.high, grid_resolution).T
        )
        actions = np.stack(action_mesh, axis=-1).reshape(-1, action_space.shape[0])

        grid_policies.append(
            agents.ConstantAgent(action=actions, policy=policies[primitive.idx_policy])
        )

    grid_policies.append(policies[-1])

    return grid_policies


def evaluate_critic_functions(
    planner: planners.Planner,
    action_skeleton: Sequence[envs.Primitive],
    env: envs.Env,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_policies = create_grid_policies(
        env, planner.policies, action_skeleton, grid_resolution
    )

    primitive = action_skeleton[0]
    env.set_primitive(primitive)
    observation = env.get_observation()
    with torch.no_grad():
        t_observation = torch.from_numpy(observation).to(planner.device)
        assert isinstance(grid_policies[0].actor.constant, torch.Tensor)
        states, actions, p_transitions = planner.dynamics.rollout(
            observation=t_observation,
            action_skeleton=action_skeleton,
            policies=grid_policies,
            batch_size=grid_policies[0].actor.constant.shape[0],
            time_index=True,
        )

        q_values = torch.zeros_like(p_transitions)
        for t, primitive in enumerate(action_skeleton):
            policy_state = planner.dynamics.decode(states[:, t], primitive)
            dim_action = torch.sum(~torch.isnan(actions[0, t])).cpu().item()
            assert isinstance(dim_action, int)
            action = actions[:, t, :dim_action]
            q_values[:, t] = planner.policies[primitive.idx_policy].critic.predict(
                policy_state, action
            )

    return q_values.cpu().numpy(), actions.cpu().numpy()


def plot_critic_functions(
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
    actions: np.ndarray,
    p_success: np.ndarray,
    rewards: np.ndarray,
    grid_q_values: np.ndarray,
    grid_actions: np.ndarray,
    path: Union[str, pathlib.Path],
    title: Optional[str] = None,
) -> None:
    def tick_labels(value: float, pos: float, dim: int) -> str:
        x = np.array(action_skeleton[0].action_space.low)
        x[dim] = value
        x = spaces.transform(
            x,
            from_space=action_skeleton[0].action_space,
            to_space=action_skeleton[0].action_scale,
        )[dim]
        return f"{x:0.2f}"

    def plot_trisurf(
        ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, **kwargs
    ) -> None:
        ax.plot_trisurf(xs, ys, zs, cmap="plasma", linewidth=0, **kwargs)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("theta [rad]")

        action_space = action_skeleton[0].action_space
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
    for t, primitive in enumerate(action_skeleton):
        ax = axes[t]

        plot_trisurf(ax, xs, ys, grid_q_values[:, t])
        ax.set_title(f"{primitive} Q(s, a)")
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
