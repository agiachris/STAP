import argparse
import pathlib

from temporal_policies import envs, planners


def visualize():
    # from temporal_policies.envs.pybox2d import visualization
    # with torch.no_grad():
    #     base_env = env.envs[0]
    #     actions, action_dims = base_env._interp_actions(20, [0, 1])
    #     actions = torch.from_numpy(actions).to(device)
    #
    #     observation = torch.from_numpy(observation).to(device)
    #     latent = planner.policies[0].encoder(observation)
    #     latent = latent.repeat(actions.shape[0], 1)
    #
    #     print(actions.shape, latent.shape)
    #     values = planner.policies[0].critic(latent, actions)
    #     values = torch.min(values[0], values[1])
    #     # values = torch.exp(values)
    #
    #     state = planner.dynamics.encode(observation, 0)
    #     state = state.repeat(actions.shape[0], 1)
    #     next_state = planner.dynamics.forward(state, 0, actions)
    #
    #     next_latent = planner.dynamics.decode(next_state, 1)
    #     next_actions = planner.policies[1].actor.predict(next_latent)
    #     next_values = planner.policies[1].critic(next_latent, next_actions)
    #     next_values = torch.min(next_values[0], next_values[1])
    #     # next_values = torch.exp(next_values)
    #
    #     visualizer = visualization.Box2DVisualizer(base_env)
    #
    #     x = actions[:, 0].cpu().numpy()
    #     y = actions[:, 1].cpu().numpy()
    #     xticks = action_dims[:, 0]
    #     yticks = action_dims[:, 1]
    #     curr_z = values.cpu().numpy()
    #     next_z = next_values.cpu().numpy()
    #
    #     visualizer.plot_xdim_theta_3d(
    #         x=x,
    #         y=y,
    #         z=[curr_z, next_z],
    #         labels=["PlaceRight", "PushLeft"],
    #         xticks=xticks,
    #         yticks=yticks,
    #         path="example_3d.png",
    #         mode="prod",
    #     )

    # print(eval_actions)
    pass


def main(args: argparse.Namespace) -> None:
    env_factory = envs.EnvFactory(args.env_config)
    planner = planners.load(
        planner_config=args.config,
        env_factory=env_factory,
        policy_checkpoints=args.policy_checkpoints,
        dynamics_checkpoint=args.dynamics_checkpoint,
        device=args.device,
    )

    action_skeleton = [(0, None), (1, None)]

    for i in range(args.num_eval):
        env = env_factory()
        state = env.get_state()
        observation = env.get_observation(action_skeleton[0][0])

        actions, predicted_success = planner.plan(observation, action_skeleton)

        rewards = planners.evaluate_plan(
            env,
            action_skeleton,
            state,
            actions,
            gif_path=pathlib.Path(f"plots/{i}.gif"),
        )
        print("success:", rewards.prod())
        print("predicted_success:", predicted_success)
        print(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "--planner_config", "--planner", "-c", help="Path to planner config"
    )
    parser.add_argument("--env_config", "--env", "-e", help="Path to env config")
    parser.add_argument(
        "--policy-checkpoints", "-p", nargs="+", help="Policy checkpoints"
    )
    parser.add_argument("--dynamics-checkpoint", "-d", help="Dynamics checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument(
        "--num-eval", "-n", type=int, default=1, help="Number of eval iterations"
    )
    args = parser.parse_args()

    main(args)
