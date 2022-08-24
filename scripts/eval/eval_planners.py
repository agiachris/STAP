import argparse
import pathlib
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import tqdm

from temporal_policies import agents, dynamics, envs, planners
from temporal_policies.envs.pybullet.table import primitives as table_primitives
from temporal_policies.utils import random, recording, timing

import eval_pybox2d_planners as pybox2d


def scale_actions(
    actions: np.ndarray,
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
) -> np.ndarray:
    scaled_actions = actions.copy()
    for t, primitive in enumerate(action_skeleton):
        action_dims = primitive.action_space.shape[0]
        scaled_actions[..., t, :action_dims] = primitive.scale_action(
            actions[..., t, :action_dims]
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
    gui: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    timer = timing.Timer()

    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)

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

    if isinstance(env, envs.pybox2d.Sequential2D):
        action_skeleton = [
            env.get_primitive_info("PlaceRight"),
            env.get_primitive_info("PushLeft"),
        ]
    elif isinstance(env, envs.pybullet.TableEnv):
        action_skeleton = [
            env.get_primitive_info(action_call) for action_call in env.action_skeleton
        ]

    num_success = 0
    pbar = tqdm.tqdm(range(num_eval), f"Evaluate {path.name}", dynamic_ncols=True)
    for i in pbar:
        if seed is not None:
            random.seed(i)

        if isinstance(planner.dynamics, dynamics.OracleDynamics):
            planner.dynamics.reset_cache()
        for policy in planner.policies:
            if isinstance(policy, agents.OracleAgent):
                policy.reset_cache()
        observation = env.reset()
        assert isinstance(observation, np.ndarray)
        state = env.get_state()

        if verbose:
            env.record_start("timelapse", mode="timelapse")

        timer.tic("planner")
        plan = planner.plan(observation, action_skeleton)
        t_planner = timer.toc("planner")

        env.record_save(path / f"planning_{i}.gif")

        rewards = planners.evaluate_plan(
            env, action_skeleton, state, plan.actions, gif_path=path / f"exec_{i}.gif"
        )
        if rewards.prod() > 0:
            num_success += 1
        pbar.set_postfix(
            dict(success=rewards.prod(), **{f"r{t}": r for t, r in enumerate(rewards)})
        )

        if verbose:
            print("success:", rewards.prod(), rewards)
            print("predicted success:", plan.p_success, plan.values)
            for primitive, action in zip(action_skeleton, plan.actions):
                if isinstance(primitive, table_primitives.Primitive):
                    primitive_action = str(primitive.Action(action))
                    primitive_action = primitive_action.replace("\n", "\n  ")
                    print(
                        "-", primitive, primitive_action[primitive_action.find("{") :]
                    )
                else:
                    print("-", primitive, action)
            print("time:", t_planner)

        if isinstance(env, envs.pybox2d.Sequential2D):
            env.set_state(state)
            (grid_q_values, grid_actions) = pybox2d.evaluate_critic_functions(
                planner, action_skeleton, env, grid_resolution
            )
            pybox2d.plot_critic_functions(
                env=env,
                action_skeleton=action_skeleton,
                actions=plan.visited_actions,
                p_success=plan.p_visited_success,
                rewards=rewards,
                grid_q_values=grid_q_values,
                grid_actions=grid_actions,
                path=path / f"values_{i}.png",
                title=f"{pathlib.Path(config).stem}: {t_planner:0.2f}s",
            )
        elif isinstance(env, envs.pybullet.TableEnv):
            if not isinstance(planner.dynamics, dynamics.OracleDynamics):
                recorder = recording.Recorder()
                recorder.start()
                env.set_state(state)
                for primitive, predicted_state, action in zip(
                    action_skeleton, plan.states[1:], plan.actions
                ):
                    env.set_primitive(primitive)
                    env._recording_text = (
                        "Action: ["
                        + ", ".join(
                            [f"{a:.2f}" for a in primitive.scale_action(action)]
                        )
                        + "]"
                    )

                    recorder.add_frame(frame=env.render())
                    env.set_observation(predicted_state)
                    recorder.add_frame(frame=env.render())
                recorder.stop()
                recorder.save(path / f"predicted_trajectory_{i}.gif")

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
                "actions": plan.actions,
                "states": plan.states,
                "scaled_actions": scale_actions(plan.actions, env, action_skeleton),
                "p_success": plan.p_success,
                "values": plan.values,
                "rewards": rewards,
                # "visited_actions": plan.visited_actions,
                # "scaled_visited_actions": scale_actions(
                #     plan.visited_actions, env, action_skeleton
                # ),
                # "visited_states": plan.visited_states,
                "p_visited_success": plan.p_visited_success,
                # "visited_values": plan.visited_values,
                "t_planner": t_planner,
            }
            if isinstance(env, envs.pybox2d.Sequential2D):
                save_dict["grid_q_values"] = grid_q_values
                save_dict["grid_actions"] = grid_actions
            np.savez_compressed(f, **save_dict)  # type: ignore

    print("Successes:", num_success, "/", num_eval)


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
    parser.add_argument("--scod-checkpoints", "-s", nargs="+", help="SCOD checkpoints")
    parser.add_argument("--dynamics-checkpoint", "-d", help="Dynamics checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    parser.add_argument("--path", default="plots", help="Path for output plots")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=40,
        help="Resolution of critic function plot",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    args = parser.parse_args()

    main(args)
