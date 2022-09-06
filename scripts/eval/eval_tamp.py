import argparse
import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np
import symbolic
import tqdm

from temporal_policies import dynamics, envs, planners
from temporal_policies.envs.pybullet.table import primitives as table_primitives
from temporal_policies.utils import recording, timing


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


def eval_tamp(
    planner_config: Union[str, pathlib.Path],
    env_config: Union[str, pathlib.Path, Dict[str, Any]],
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]],
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]],
    device: str,
    num_eval: int,
    path: Union[str, pathlib.Path],
    pddl_domain: str,
    pddl_problem: str,
    max_depth: int = 5,
    timeout: float = 10.0,
    verbose: bool = False,
    seed: Optional[int] = None,
    gui: Optional[int] = None,
) -> None:
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
    for i in tqdm.tqdm(range(num_eval)):
        timer = timing.Profiler()

        # Initialize environment.
        observation, info = env.reset(seed=seed if i == 0 else None)
        seed = info["seed"]
        state = env.get_state()

        task_plans = []
        motion_plans = []

        # Task planning outer loop.
        action_skeleton_generator = task_plan(
            pddl=pddl, env=env, max_depth=max_depth, timeout=timeout, verbose=verbose
        )
        timer.tic("task_planner")
        for action_skeleton in action_skeleton_generator:
            timer.toc("task_planner")

            # Motion planning inner loop.
            timer.tic("motion_planner")
            motion_plan = planner.plan(observation, action_skeleton)
            timer.toc("motion_planner")

            # Reset env for oracle value/dynamics.
            env.set_state(state)

            task_plans.append(action_skeleton)
            motion_plans.append(motion_plan)

            timer.tic("task_planner")
        timer.toc("task_planner")

        # Get best TAMP plan.
        idx_best = np.argmax([plan.p_success for plan in motion_plans])

        rewards = planners.evaluate_plan(
            env=env,
            action_skeleton=task_plans[idx_best],
            state=state,
            actions=motion_plans[idx_best].actions,
            gif_path=path / f"exec_{i}.gif",
        )
        # for idx_plan in range(len(task_plans)):
        #     if idx_plan == idx_best:
        #         continue
        #     planners.evaluate_plan(
        #         env=env,
        #         action_skeleton=task_plans[idx_plan],
        #         state=state,
        #         actions=motion_plans[idx_plan].actions,
        #         gif_path=path / f"exec_{i}-{idx_plan}.gif",
        #     )

        t_task_planner = timer.compute_sum("task_planner")
        t_motion_planner = timer.compute_sum("motion_planner")

        # Print planning results.
        if verbose:
            print("success:", rewards.prod(), rewards)
            print(
                "predicted success:",
                motion_plans[idx_best].p_success,
                motion_plans[idx_best].values,
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
            print("task planning time:", t_task_planner)
            print("motion planning time:", t_motion_planner)
            continue

        # Record imagined trajectory.
        if isinstance(env, envs.pybullet.TableEnv):
            if not isinstance(planner.dynamics, dynamics.OracleDynamics):
                recorder = recording.Recorder()
                recorder.start()
                env.set_state(state)
                for primitive, predicted_state, action in zip(
                    task_plans[idx_best],
                    motion_plans[idx_best].states[1:],
                    motion_plans[idx_best].actions,
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
                recorder.save(path / "predicted_trajectory_{i}.gif")

        # Save planning results.
        with open(path / f"results_{i}.npz", "wb") as f:
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
                "action_skeleton": task_plans[idx_best],
                "actions": motion_plans[idx_best].actions,
                "states": motion_plans[idx_best].states,
                "scaled_actions": scale_actions(
                    motion_plans[idx_best].actions, env, task_plans[idx_best]
                ),
                "p_success": motion_plans[idx_best].p_success,
                "values": motion_plans[idx_best].values,
                "rewards": rewards,
                "visited_actions": motion_plans[idx_best].visited_actions,
                "scaled_visited_actions": scale_actions(
                    motion_plans[idx_best].visited_actions, env, action_skeleton
                ),
                "visited_states": motion_plans[idx_best].visited_states,
                "p_visited_success": motion_plans[idx_best].p_visited_success,
                "visited_values": motion_plans[idx_best].visited_values,
                "t_task_planner": t_task_planner,
                "t_motion_planner": t_motion_planner,
                "seed": seed,
                "discarded": [
                    {
                        "action_skeleton": task_plans[i],
                        "actions": motion_plans[i].actions,
                        "states": motion_plans[i].states,
                        "scaled_actions": scale_actions(
                            motion_plans[i].actions, env, task_plans[i]
                        ),
                        "p_success": motion_plans[i].p_success,
                        "values": motion_plans[i].values,
                        "visited_actions": motion_plans[i].visited_actions,
                        "scaled_visited_actions": scale_actions(
                            motion_plans[i].visited_actions, env, action_skeleton
                        ),
                        "visited_states": motion_plans[i].visited_states,
                        "p_visited_success": motion_plans[i].p_visited_success,
                        "visited_values": motion_plans[i].visited_values,
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
