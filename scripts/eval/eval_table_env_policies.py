#!/usr/bin/env python3

import argparse
import pathlib
from typing import Dict, Optional, List, Tuple, Union

from ctrlutils import eigen
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch

from temporal_policies import agents, envs
from temporal_policies.envs.pybullet.sim import math
from temporal_policies.envs.pybullet.table import (
    objects,
    object_state,
    primitives,
    primitive_actions,
)
from temporal_policies.envs.pybullet.sim.robot import ControlException
from temporal_policies.utils import random

import pybullet as p


def evaluate_critic(
    policy: agents.Agent,
    observations: np.ndarray,
    actions: np.ndarray,
    policy_args: Optional[Dict[str, List[int]]],
) -> np.ndarray:
    with torch.no_grad():
        t_observations = torch.from_numpy(observations).to(policy.device)
        t_actions = torch.from_numpy(actions).to(policy.device)
        t_states = policy.encoder.encode(t_observations, policy_args)
        t_q_values = policy.critic.predict(t_states, t_actions)
        q_values = t_q_values.cpu().numpy()

    return q_values


def evaluate_pick_critic_state(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Pick,
    policy: agents.Agent,
    observation: np.ndarray,
    action: np.ndarray,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    policy_args = primitive.get_policy_args()
    assert policy_args is not None
    idx_object = policy_args["observation_indices"][1]  # First pick arg.
    # Create grid of xy positions on the table.
    xy_min = np.array(env.observation_space.low[idx_object, :2])
    xy_max = np.array(env.observation_space.high[idx_object, :2])
    xy_min[1] = max(-0.45, xy_min[1])
    xy_max[1] = min(0.45, xy_max[1])
    z = primitive.arg_objects[0].size[2] / 2
    xs, ys = np.meshgrid(*np.linspace(xy_min, xy_max, grid_resolution).T)

    # Create observation batch.
    observations = np.tile(
        observation, (xs.size, *np.ones_like(env.observation_space.shape))
    )
    obs = object_state.ObjectState(observations)
    obs.pos[:, idx_object, 0] = xs.flatten()
    obs.pos[:, idx_object, 1] = ys.flatten()
    obs.pos[:, idx_object, 2] = z

    # Create action batch.
    actions = np.tile(
        action, (observations.shape[0], *([1] * len(env.action_space.shape)))
    )
    normalized_actions = primitives.Pick.normalize_action(actions)

    q_values = evaluate_critic(
        policy, observations, normalized_actions, primitive.get_policy_args()
    )

    return q_values, observations[:, idx_object, :2]


def evaluate_pick_critic_action(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Pick,
    policy: agents.Agent,
    observation: np.ndarray,
    action: np.ndarray,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Create grid of xy pick positions.
    xy_min = np.array(primitives.Pick.action_scale.low[:2])
    xy_max = np.array(primitives.Pick.action_scale.high[:2])
    xs, ys = np.meshgrid(*np.linspace(xy_min, xy_max, grid_resolution).T)

    # Create action batch.
    actions = np.tile(action, (xs.size, *([1] * len(env.action_space.shape))))
    a = primitive_actions.PickAction(actions)
    a.pos[:, 0] = xs.flatten()
    a.pos[:, 1] = ys.flatten()
    normalized_actions = primitives.Pick.normalize_action(actions)

    # Create observation batch.
    observations = np.tile(
        observation, (xs.size, *([1] * len(env.observation_space.shape)))
    )

    q_values = evaluate_critic(
        policy, observations, normalized_actions, primitive.get_policy_args()
    )

    # Compute image coordinates (on top of target object).
    policy_args = primitive.get_policy_args()
    assert policy_args is not None
    idx_object = policy_args["observation_indices"][1]  # First pick arg.
    obs = object_state.ObjectState(observation)
    xy = obs.pos[idx_object, :2]
    theta = obs.aa[idx_object, 2]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    absolute_actions = a.pos[:, :2] @ R.T + xy

    return q_values, absolute_actions


def evaluate_place_critic_action(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Place,
    policy: agents.Agent,
    observation: np.ndarray,
    action: np.ndarray,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Create grid of xy place positions.
    xy_min = np.array(primitives.Place.action_scale.low[:2])
    xy_max = np.array(primitives.Place.action_scale.high[:2])
    # xy_min[1] = max(-0.45, xy_min[1])
    # xy_max[1] = min(0.45, xy_max[1])
    xs, ys = np.meshgrid(*np.linspace(xy_min, xy_max, grid_resolution).T)

    # Create action batch.
    actions = np.tile(action, (xs.size, *([1] * len(env.action_space.shape))))
    a = primitive_actions.PlaceAction(actions)
    a.pos[:, 0] = xs.flatten()
    a.pos[:, 1] = ys.flatten()
    normalized_actions = primitives.Place.normalize_action(actions)

    # Create observation batch.
    observations = np.tile(
        observation, (xs.size, *([1] * len(env.observation_space.shape)))
    )

    q_values = evaluate_critic(
        policy, observations, normalized_actions, primitive.get_policy_args()
    )

    # Compute image coordinates (on top of target object).
    policy_args = primitive.get_policy_args()
    assert policy_args is not None
    idx_object = policy_args["observation_indices"][2]  # Second place arg.
    obs = object_state.ObjectState(observation)
    xy = obs.pos[idx_object, :2]
    theta = obs.aa[idx_object, 2]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    absolute_actions = actions[:, :2] @ R.T + xy

    return q_values, absolute_actions


def evaluate_pick_state(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Pick,
    policy: agents.Agent,
    path: pathlib.Path,
    grid_resolution: int,
) -> None:
    grid_q_values, grid_states = evaluate_pick_critic_state(
        env=env,
        primitive=primitive,
        policy=policy,
        observation=env.get_observation(),
        action=primitive.sample_action().vector,
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


def evaluate_pick_action(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Pick,
    policy: agents.Agent,
    observation: np.ndarray,
    z: float,
    theta: float,
    path: pathlib.Path,
    grid_resolution: int,
) -> None:
    grid_q_values, grid_states = evaluate_pick_critic_action(
        env=env,
        primitive=primitive,
        policy=policy,
        observation=observation,
        action=primitive_actions.PickAction(
            pos=np.array([0.0, 0.0, z]), theta=theta
        ).vector,
        grid_resolution=grid_resolution,
    )
    grid_q_values = grid_q_values.clip(0.0, 1.0)

    obs = object_state.ObjectState(observation)
    try:
        env.robot.goto_pose(
            pos=np.array(
                [obs.pos[0, 0], -obs.pos[0, 1], env.robot.home_pose.pos[2] + z]
            ),
            quat=primitives.compute_top_down_orientation(
                theta,
                eigen.Quaterniond(primitive.arg_objects[0].pose().quat),
            ),
        )
    except ControlException:
        pass

    path_obj = save_critic_obj(
        grid_states,
        grid_q_values,
        path=path / "assets",
        name=f"action_q_values_z{z:.1f}_th{theta:.2f}",
        grid_resolution=grid_resolution,
        z_scale=0.0,
        z_height=(primitive.arg_objects[0].aabb()[1, 2]) + 0.001,
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


def evaluate_place_action(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Place,
    policy: agents.Agent,
    observation: np.ndarray,
    action: np.ndarray,
    path: pathlib.Path,
    grid_resolution: int,
) -> None:
    grid_q_values, grid_states = evaluate_place_critic_action(
        env=env,
        primitive=primitive,
        policy=policy,
        observation=observation,
        action=action,
        grid_resolution=grid_resolution,
    )
    grid_q_values = grid_q_values.clip(0.0, 1.0)

    path_obj = save_critic_obj(
        grid_states,
        grid_q_values,
        path=path / "assets",
        name="action_q_values",
        grid_resolution=grid_resolution,
        z_scale=0.0,
        z_height=0.001,
    )

    for view in ("front", "top"):
        plot_critic_overlay(
            env=env,
            path_obj=path_obj,
            path=path,
            name=f"action_values_{view}",
            opacity=0.9,
            view=view,
        )


def evaluate_pick(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Pick,
    policy: agents.Agent,
    path: pathlib.Path,
    grid_resolution: int,
) -> None:
    def _evaluate_pick(path: pathlib.Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Setup scene.
        env.reset()
        obj = primitive.arg_objects[0]
        obj.set_pose(math.Pose(pos=np.array([0.4, 0.0, -obj.bbox[0, 2]])))
        if "rack" in env.objects:
            rack = env.objects["rack"]
            rack.set_pose(math.Pose(pos=np.array([0.5, 0.25, -rack.bbox[0, 2]])))

        evaluate_pick_state(
            env=env,
            primitive=primitive,
            policy=policy,
            path=path,
            grid_resolution=grid_resolution,
        )

        observation = env.get_observation()
        for theta in np.linspace(0, np.pi / 2, 3):
            evaluate_pick_action(
                env=env,
                primitive=primitive,
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
                primitive=primitive,
                policy=policy,
                observation=observation,
                z=z,
                theta=0.0,
                path=path,
                grid_resolution=grid_resolution,
            )

    obj = primitive.arg_objects[0]
    if isinstance(obj, objects.Variant):
        for idx_variant in range(len(obj.variants)):
            obj.set_variant(idx_variant, [primitive], lock=True)
            _evaluate_pick(path / f"{primitive}-{type(obj.body).__name__.lower()}")
    else:
        _evaluate_pick(path / str(primitive))


def evaluate_place(
    env: envs.pybullet.TableEnv,
    primitive: primitives.Place,
    policy: agents.Agent,
    path: pathlib.Path,
    grid_resolution: int,
):
    def _evaluate_place(path: pathlib.Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Setup scene.
        env.reset()
        if "rack" in env.objects:
            rack = env.objects["rack"]
            rack.set_pose(math.Pose(pos=np.array([0.5, 0.25, -rack.bbox[0, 2]])))

        evaluate_place_action(
            env=env,
            primitive=primitive,
            policy=policy,
            observation=env.get_observation(),
            action=primitive.sample_action().vector,
            path=path,
            grid_resolution=grid_resolution,
        )

    obj = primitive.arg_objects[0]
    if isinstance(obj, objects.Variant):
        for idx_variant in range(len(obj.variants)):
            obj.set_variant(idx_variant, [primitive], lock=True)
            _evaluate_place(path / f"{primitive}-{type(obj.body).__name__.lower()}")
    else:
        _evaluate_place(path / str(primitive))


def evaluate_policies(
    env_config: Optional[Union[str, pathlib.Path]],
    checkpoint: Union[str, pathlib.Path],
    device: str,
    path: Union[str, pathlib.Path],
    grid_resolution: int,
    verbose: bool,
    seed: Optional[int] = None,
    eval_results: Optional[str] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    path = pathlib.Path(path)

    if eval_results is not None:
        eval_env_config = pathlib.Path(checkpoint).parent / "eval/env_config.yaml"
        eval_env = envs.load(eval_env_config)
        policy = agents.load(checkpoint=checkpoint, device=device)
        assert isinstance(policy, agents.RLAgent)

        assert isinstance(eval_env, envs.pybullet.TableEnv)
        with open(eval_results, "rb") as f:
            seed = int(np.load(f, allow_pickle=True)["seed"])
        observation, info = eval_env.reset(seed=seed)
        action = (
            policy.actor.predict(
                policy.encoder.encode(
                    torch.from_numpy(observation).to(policy.device), info["policy_args"]
                )
            )
            .cpu()
            .detach()
            .numpy()
        )
        print(action)
        eval_env.record_start(frequency=10)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        print(reward, terminated, truncated)
    else:
        if env_config is not None:
            env = envs.load(env_config)

        policy = agents.load(checkpoint=checkpoint, device=device)
        assert isinstance(policy, agents.RLAgent)

        if env_config is None:
            env = policy.env

        assert isinstance(env, envs.pybullet.TableEnv)
        primitive = env.get_primitive()
        if isinstance(primitive, primitives.Pick):
            evaluate_pick(
                env=env,
                primitive=primitive,
                policy=policy,
                path=path,
                grid_resolution=grid_resolution,
            )
        elif isinstance(primitive, primitives.Place):
            evaluate_place(
                env=env,
                primitive=primitive,
                policy=policy,
                path=path,
                grid_resolution=grid_resolution,
            )


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

    env.render_mode = f"{view}_high_res"
    img_rgb = env.render()
    img = PIL.Image.fromarray(img_rgb)
    img.save(path / f"{name}.png")

    p.removeBody(body_id, physicsClientId=env.physics_id)


def main(args: argparse.Namespace) -> None:
    evaluate_policies(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--checkpoint", help="Policy checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--eval-results", type=str, help="Path to results_i.npz file.")
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=40,
        help="Resolution of critic function plot",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    args = parser.parse_args()

    main(args)
