#!/usr/bin/env python3

import argparse
import pathlib
from typing import Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch

from temporal_policies import agents, envs
from temporal_policies.envs.pybullet.table import objects, object_state, primitives
from temporal_policies.envs.pybullet.sim.robot import ControlException
from temporal_policies.utils import random

import pybullet as p


def evaluate_pick_critic_state(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Pick,
    policy: agents.Agent,
    observation: np.ndarray,
    action: np.ndarray,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xy_min = np.array(env.observation_space.low[:2])
    xy_max = np.array(env.observation_space.high[:2])
    xy_min[1] = max(-0.45, xy_min[1])
    xy_max[1] = min(0.45, xy_max[1])
    obs = object_state.ObjectState(observation)
    z = primitive.policy_args[0].size[2] / 2
    xs, ys = np.meshgrid(*np.linspace(xy_min, xy_max, grid_resolution).T)

    observations = np.tile(
        observation, (xs.size, *np.ones_like(env.observation_space.shape))
    )
    obs = object_state.ObjectState(observations)
    obs.pos[:, 0, 0] = xs.flatten()
    obs.pos[:, 0, 1] = ys.flatten()
    obs.pos[:, 0, 2] = z

    actions = np.tile(
        action, (observations.shape[0], *([1] * len(env.action_space.shape)))
    )
    normalized_actions = env.get_primitive().normalize_action(actions)

    with torch.no_grad():
        torch_observations = torch.from_numpy(observations).to(policy.device)
        torch_actions = torch.from_numpy(normalized_actions).to(policy.device)
        torch_states = policy.encoder.encode(torch_observations)
        torch_q_values = policy.critic.predict(torch_states, torch_actions)
        q_values = torch_q_values.cpu().numpy()

    return q_values, observations[:, :2]


def evaluate_pick_critic_action(
    env: envs.pybullet.TableEnv,
    policy: agents.Agent,
    observation: np.ndarray,
    action: np.ndarray,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xy_min = np.array(env.get_primitive().action_scale.low[:2])
    xy_max = np.array(env.get_primitive().action_scale.high[:2])
    xs, ys = np.meshgrid(*np.linspace(xy_min, xy_max, grid_resolution).T)

    actions = np.tile(action, (xs.size, *([1] * len(env.action_space.shape))))
    actions[:, 0] = xs.flatten()
    actions[:, 1] = ys.flatten()
    normalized_actions = env.get_primitive().normalize_action(actions)

    observations = np.tile(
        observation, (xs.size, *([1] * len(env.observation_space.shape)))
    )

    with torch.no_grad():
        torch_observations = torch.from_numpy(observations).to(policy.device)
        torch_actions = torch.from_numpy(normalized_actions).to(policy.device)
        torch_states = policy.encoder.encode(torch_observations)
        torch_q_values = policy.critic.predict(torch_states, torch_actions)
        q_values = torch_q_values.cpu().numpy()

    obs = object_state.ObjectState(observation)
    xy = obs.pos[0, :2]
    theta = obs.aa[0, 2]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    absolute_actions = actions[:, :2] @ R.T + xy

    return q_values, absolute_actions


def save_critic_obj(
    grid_states: np.ndarray,
    grid_q_values: np.ndarray,
    path: pathlib.Path,
    name: str,
    grid_resolution: int,
    z_scale: float = 0.1,
    z_height: float = 0.001,
) -> pathlib.Path:
    xs = np.reshape(grid_states[:, 0], (grid_resolution, grid_resolution))
    ys = np.reshape(grid_states[:, 1], (grid_resolution, grid_resolution))
    zs = np.reshape(grid_q_values, (grid_resolution, grid_resolution))

    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{name}.obj", "w") as f:
        f.write(f"o {name}\n")
        f.write(f"mtllib {name}.mtl\n")

        # Vertices.
        for u in range(grid_resolution):
            for v in range(grid_resolution):
                f.write(f"v {xs[u, v]} {ys[u, v]} {z_scale * zs[u, v] + z_height}\n")

        # Texture coordinates.
        for u in range(grid_resolution):
            for v in range(grid_resolution):
                f.write(
                    f"vt {v / (grid_resolution - 1)} {1 - u / (grid_resolution - 1)}\n"
                )

        # Face elements.
        f.write(f"g {name}\n")
        f.write(f"usemtl {name}\n")
        for u in range(grid_resolution - 1):
            for v in range(grid_resolution - 1):
                uv = u * grid_resolution + v + 1
                uvv = uv + 1
                uuv = uv + grid_resolution
                uuvv = uuv + 1
                f.write(f"f {uv}/{uv} {uvv}/{uvv} {uuvv}/{uuvv} {uuv}/{uuv}\n")

    with open(path / f"{name}.mtl", "w") as f:
        f.write(f"newmtl {name}\n")
        f.write("d 0.5\n")
        f.write(f"map_Kd {name}.png\n")

    cmap = plt.get_cmap("plasma")
    img_values = (255 * cmap(zs) + 0.5).astype(np.uint8)
    img = PIL.Image.fromarray(img_values)
    img.save(path / f"{name}.png")

    return path / f"{name}.obj"


def plot_critic_overlay(
    env: envs.pybullet.TableEnv,
    path_obj: pathlib.Path,
    path: pathlib.Path,
    name: str,
    opacity: float = 0.9,
    view: str = "front",
) -> None:
    visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(path_obj),
        rgbaColor=[1.0, 1.0, 1.0, opacity],
        physicsClientId=env.physics_id,
    )
    collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.6, 0.5, 0.05],
        collisionFramePosition=[0.3, 0, 0.05],
        physicsClientId=env.physics_id,
    )
    body_id = p.createMultiBody(
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        physicsClientId=env.physics_id,
    )

    img_rgb = env.render(view, resolution=(1620, 1080))
    img = PIL.Image.fromarray(img_rgb)
    img.save(path / f"{name}.png")

    p.removeBody(body_id, physicsClientId=env.physics_id)


def evaluate_pick_action(
    env: envs.pybullet.TableEnv,
    policy: agents.Agent,
    observation: np.ndarray,
    z: float,
    theta: float,
    path: pathlib.Path,
    grid_resolution: int,
):
    grid_q_values, grid_states = evaluate_pick_critic_action(
        env=env,
        policy=policy,
        observation=observation,
        action=np.array([0, 0, 0, theta], dtype=np.float32),
        grid_resolution=grid_resolution,
    )
    grid_q_values = grid_q_values.clip(0.0, 1.0)

    obs = object_state.ObjectState(observation)
    env.robot.goto_pose(
        pos=np.array([obs.pos[0, 0], -obs.pos[0, 1], env.robot.home_pose.pos[2] + z]),
        quat=primitives.compute_top_down_orientation(
            eigen.Quaterniond(env.robot.home_pose.quat),
            eigen.Quaterniond(env.get_primitive().policy_args[0].pose().quat),
            theta=theta,
        ),
    )

    path_obj = save_critic_obj(
        grid_states,
        grid_q_values,
        path=path / "assets",
        name=f"action_q_values_z{z:.1f}_th{theta:.2f}",
        grid_resolution=grid_resolution,
        z_scale=0.0,
        z_height=(obs.box_size[0, 2] if obs.box_size[0, 2] > 0 else 0.04) + 0.001,
    )

    for view in ("front", "top"):
        plot_critic_overlay(
            env=env,
            path_obj=path_obj,
            path=path,
            name=f"action_values_z{z:.2f}_th{theta:.2f}_{view}",
            opacity=0.75,
            view=view,
        )


def evaluate_policies(
    checkpoint: Union[str, pathlib.Path],
    device: str,
    path: Union[str, pathlib.Path],
    grid_resolution: int,
    verbose: bool,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    policy = agents.load(checkpoint=checkpoint, device=device)
    assert isinstance(policy, agents.RLAgent)
    env = policy.env
    assert isinstance(env, envs.pybullet.TableEnv)

    primitive = env.get_primitive()
    assert isinstance(primitive, primitives.Pick)
    if isinstance(primitive.policy_args[0], objects.Variant):
        primitive.policy_args[0].set_variant(0)

    grid_q_values, grid_states = evaluate_pick_critic_state(
        env=env,
        primitive=primitive,
        policy=policy,
        observation=env.get_observation(),
        action=primitive.sample_action(),
        grid_resolution=grid_resolution,
    )
    grid_q_values = grid_q_values.clip(0.0, 1.0)

    path_obj = save_critic_obj(
        grid_states,
        grid_q_values,
        path=path / "assets",
        name="state_q_values",
        grid_resolution=grid_resolution,
        z_scale=0,  # .05,
        z_height=0.001,
    )
    for view in ("front", "top"):
        plot_critic_overlay(
            env=env,
            path_obj=path_obj,
            path=path,
            name=f"state_values_{view}",
            opacity=0.9,
            view=view,
        )

    observation = env.reset()
    for theta in np.linspace(0, np.pi / 2, 3):
        evaluate_pick_action(
            env=env,
            policy=policy,
            observation=observation,
            z=0.0,
            theta=theta,
            path=path,
            grid_resolution=grid_resolution,
        )
    for z in np.linspace(-0.05, 0.05, 3):
        evaluate_pick_action(
            env=env,
            policy=policy,
            observation=observation,
            z=z,
            theta=0.0,
            path=path,
            grid_resolution=grid_resolution,
        )


def main(args: argparse.Namespace) -> None:
    evaluate_policies(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
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
