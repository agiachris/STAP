import argparse
import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import symbolic
import tqdm

from temporal_policies import dynamics, envs, planners
from temporal_policies.envs.pybullet.table import primitives as table_primitives
from temporal_policies.utils import recording, timing


def seed_generator(
    num_eval: int,
    path_results: Optional[Union[str, pathlib.Path]] = None,
) -> Generator[
    Tuple[
        Optional[int],
        Optional[Tuple[np.ndarray, planners.PlanningResult, Optional[List[float]]]],
    ],
    None,
    None,
]:
    if path_results is not None:
        npz_files = sorted(
            pathlib.Path(path_results).glob("results_*.npz"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        for npz_file in npz_files:
            with open(npz_file, "rb") as f:
                npz = np.load(f, allow_pickle=True)
                seed: int = npz["seed"].item()
                rewards = np.array(npz["rewards"])
                plan = planners.PlanningResult(
                    actions=np.array(npz["actions"]),
                    states=np.array(npz["states"]),
                    p_success=npz["p_success"].item(),
                    values=np.array(npz["values"]),
                )
                t_planner: List[float] = npz["t_planner"].item()

            yield seed, (rewards, plan, t_planner)

    if num_eval is not None:
        yield 0, None
        for _ in range(num_eval - 1):
            yield None, None


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


def task_plan(
    pddl: symbolic.Pddl,
    env: envs.Env,
    max_depth: int = 5,
    timeout: float = 10.0,
    verbose: bool = False,
) -> Generator[List[envs.Primitive], None, None]:
    planner = symbolic.Planner(pddl, pddl.initial_state)
    bfs = symbolic.BreadthFirstSearch(
        planner.root, max_depth=max_depth, timeout=timeout, verbose=False
    )
    for plan in bfs:
        action_skeleton = [
            env.get_primitive_info(action_call=str(node.action)) for node in plan[1:]
        ]
        yield action_skeleton


def evaluate_plan(
    idx_iter: int,
    env: envs.Env,
    planner: planners.Planner,
    action_skeleton: Sequence[envs.Primitive],
    plan: planners.PlanningResult,
    rewards: np.ndarray,
    path: pathlib.Path,
    grid_resolution: Optional[int] = None,
) -> Dict[str, Any]:
    assert isinstance(env, envs.pybullet.TableEnv)
    recorder = recording.Recorder()
    recorder.start()
    for primitive, predicted_state, action in zip(
        action_skeleton, plan.states[1:], plan.actions
    ):
        env.set_primitive(primitive)
        env._recording_text = (
            "Action: ["
            + ", ".join([f"{a:.2f}" for a in primitive.scale_action(action)])
            + "]"
        )

        recorder.add_frame(frame=env.render())
        env.set_observation(predicted_state)
        recorder.add_frame(frame=env.render())
    recorder.stop()
    recorder.save(path / f"predicted_trajectory_{idx_iter}.gif")

    return {}


def eval_tamp(
    planner_config: Union[str, pathlib.Path],
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    closed_loop: int,
    pddl_domain: str,
    pddl_problem: str,
    max_depth: int = 5,
    timeout: float = 10.0,
    verbose: bool = False,
    load_path: Optional[Union[str, pathlib.Path]] = None,
    seed: Optional[int] = None,
    gui: Optional[int] = None,
) -> None:
    timer = timing.Timer()

    # Load environment.
    env_kwargs = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    env_factory = envs.EnvFactory(config=env_config)
    env = env_factory(**env_kwargs)

    # Load planner.
    planner = planners.load(
        config=planner_config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        scod_checkpoints=scod_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    path = pathlib.Path(path) / pathlib.Path(planner_config).stem
    path.mkdir(parents=True, exist_ok=True)

    # Load pddl.
    pddl = symbolic.Pddl(pddl_domain, pddl_problem)

    # Run TAMP.
    num_success = 0
    pbar = tqdm.tqdm(
        seed_generator(num_eval, load_path), f"Evaluate {path.name}", dynamic_ncols=True
    )
    for idx_iter, (seed, loaded_plan) in enumerate(pbar):
        # Initialize environment.
        observation, info = env.reset(seed=seed)
        seed = info["seed"]
        state = env.get_state()

        task_plans = []
        motion_plans = []
        motion_planner_times = []

        # Task planning outer loop.
        action_skeleton_generator = task_plan(
            pddl=pddl, env=env, max_depth=max_depth, timeout=timeout, verbose=verbose
        )
        for action_skeleton in action_skeleton_generator:
            timer.tic("motion_planner")
            env.set_primitive(action_skeleton[0])
            plan = planner.plan(env.get_observation(), action_skeleton)
            t_motion_planner = timer.toc("motion_planner")

            task_plans.append(action_skeleton)
            motion_plans.append(plan)
            motion_planner_times.append(t_motion_planner)

            # Reset env for oracle value/dynamics.
            if isinstance(planner.dynamics, dynamics.OracleDynamics):
                env.set_state(state)

            if "greedy" in str(planner_config):
                break

        # Get best TAMP plan.
        if motion_plans[0].visited_values is not None:
            values = [(plan.p_success, -plan.visited_values[0]) for plan in motion_plans]
            best = max(values)
            idx_best = values.index(best)
        else:
            idx_best = np.argmax([plan.p_success for plan in motion_plans])
        best_task_plan = task_plans[idx_best]
        best_motion_plan = motion_plans[idx_best]

        if closed_loop:
            env.record_start()

            # Execute one step.
            action = best_motion_plan.actions[0]
            assert isinstance(action, np.ndarray)
            state = best_motion_plan.states[0]
            assert isinstance(state, np.ndarray)
            p_success = best_motion_plan.p_success
            value = best_motion_plan.values[0]
            env.set_primitive(best_task_plan[0])
            _, reward, _, _, _ = env.step(action)

            # Run closed-loop planning on the rest.
            rewards, plan, t_planner = planners.run_closed_loop_planning(
                env, best_task_plan[1:], planner, timer
            )
            rewards = np.append(reward, rewards)
            plan = planners.PlanningResult(
                actions=np.concatenate((action[None, ...], plan.actions), axis=0),
                states=np.concatenate((state[None, ...], plan.states), axis=0),
                p_success=p_success * plan.p_success,
                values=np.append(value, plan.values),
            )
            env.record_stop()

            # Save recording.
            gif_path = path / f"planning_{idx_iter}.gif"
            if (rewards == 0.0).any():
                gif_path = gif_path.parent / f"{gif_path.name}_fail{gif_path.suffix}"
            env.record_save(gif_path, reset=True)

            if t_planner is not None:
                motion_planner_times += t_planner
        else:
            # Execute plan.
            rewards = planners.evaluate_plan(
                env,
                task_plans[idx_best],
                motion_plans[idx_best].actions,
                gif_path=path / f"planning_{idx_iter}.gif",
            )

        if rewards.prod() > 0:
            num_success += 1
        pbar.set_postfix(
            dict(
                success=rewards.prod(),
                **{f"r{t}": r for t, r in enumerate(rewards)},
                num_successes=f"{num_success} / {idx_iter + 1}",
            )
        )

        # Print planning results.
        if verbose:
            print("success:", rewards.prod(), rewards)
            print(
                "predicted success:",
                motion_plans[idx_best].p_success,
                motion_plans[idx_best].values,
            )
            if closed_loop:
                print(
                    "visited predicted success:",
                    motion_plans[idx_best].p_visited_success,
                    motion_plans[idx_best].visited_values,
                )
            for primitive, action in zip(
                task_plans[idx_best], motion_plans[idx_best].actions
            ):
                if isinstance(primitive, table_primitives.Primitive):
                    primitive_action = str(primitive.Action(action))
                    primitive_action = primitive_action.replace("\n", "\n  ")
                    print(
                        "-", primitive, primitive_action[primitive_action.find("{") :]
                    )
                else:
                    print("-", primitive, action)
            print("discarded plans:")
            for idx_plan in range(len(task_plans)):
                if idx_plan == idx_best:
                    continue
                print(
                    "  - predicted success:",
                    motion_plans[idx_plan].p_success,
                    motion_plans[idx_plan].values,
                )
                for primitive, action in zip(
                    task_plans[idx_plan], motion_plans[idx_plan].actions
                ):
                    if isinstance(primitive, table_primitives.Primitive):
                        primitive_action = str(primitive.Action(action))
                        primitive_action = primitive_action.replace("\n", "\n      ")
                        print(
                            "    -",
                            primitive,
                            primitive_action[primitive_action.find("{") :],
                        )
                    else:
                        print("    -", primitive, action)
            continue

        # Save planning results.
        with open(path / f"results_{idx_iter}.npz", "wb") as f:
            save_dict = {
                "args": {
                    "planner_config": planner_config,
                    "env_config": env_config,
                    "policy_checkpoints": policy_checkpoints,
                    "dynamics_checkpoint": dynamics_checkpoint,
                    "device": device,
                    "num_eval": num_eval,
                    "path": path,
                    "pddl_domain": pddl_domain,
                    "pddl_problem": pddl_problem,
                    "max_depth": max_depth,
                    "timeout": timeout,
                    "seed": seed,
                    "verbose": verbose,
                },
                "observation": observation,
                "state": state,
                "action_skeleton": list(map(str, task_plans[idx_best])),
                "actions": motion_plans[idx_best].actions,
                "states": motion_plans[idx_best].states,
                "scaled_actions": scale_actions(
                    motion_plans[idx_best].actions, env, task_plans[idx_best]
                ),
                "p_success": motion_plans[idx_best].p_success,
                "values": motion_plans[idx_best].values,
                "rewards": rewards,
                # "visited_actions": motion_plans[idx_best].visited_actions,
                # "scaled_visited_actions": scale_actions(
                #     motion_plans[idx_best].visited_actions, env, action_skeleton
                # ),
                # "visited_states": motion_plans[idx_best].visited_states,
                "p_visited_success": motion_plans[idx_best].p_visited_success,
                # "visited_values": motion_plans[idx_best].visited_values,
                # "t_task_planner": t_task_planner,
                # "t_motion_planner": t_motion_planner,
                "t_planner": motion_planner_times,
                "seed": seed,
                "discarded": [
                    {
                        "action_skeleton": list(map(str, task_plans[i])),
                        # "actions": motion_plans[i].actions,
                        # "states": motion_plans[i].states,
                        # "scaled_actions": scale_actions(
                        #     motion_plans[i].actions, env, task_plans[i]
                        # ),
                        "p_success": motion_plans[i].p_success,
                        # "values": motion_plans[i].values,
                        # "visited_actions": motion_plans[i].visited_actions,
                        # "scaled_visited_actions": scale_actions(
                        #     motion_plans[i].visited_actions, env, action_skeleton
                        # ),
                        # "visited_states": motion_plans[i].visited_states,
                        "p_visited_success": motion_plans[i].p_visited_success,
                        # "visited_values": motion_plans[i].visited_values,
                    }
                    for i in range(len(task_plans))
                    if i != idx_best
                ],
            }
            np.savez_compressed(f, **save_dict)  # type: ignore


def main(args: argparse.Namespace) -> None:
    eval_tamp(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-config", "--planner", "-c", help="Path to planner config"
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
    parser.add_argument(
        "--closed-loop", default=1, type=int, help="Run closed-loop planning"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--pddl-domain", help="Pddl domain")
    parser.add_argument("--pddl-problem", help="Pddl problem")
    parser.add_argument(
        "--max-depth", type=int, default=4, help="Task planning search depth"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Task planning timeout"
    )
    args = parser.parse_args()

    main(args)
