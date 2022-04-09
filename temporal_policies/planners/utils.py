import functools
import pathlib
from typing import Any, Dict, Optional, Iterable, Sequence, Tuple, Union

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import agents, dynamics, envs, planners
from temporal_policies.utils import configs


class PlannerFactory(configs.Factory):
    """Planner factory."""

    def __init__(
        self,
        planner_config: Union[str, pathlib.Path, Dict[str, Any]],
        env_factory: envs.EnvFactory,
        policy_checkpoints: Optional[Sequence[Optional[str]]] = None,
        dynamics_checkpoint: Optional[str] = None,
    ):
        """Creates the planner factory from a planner_config.

        Args:
            planner_config: Planner config path or dict.
            env_factory: Env factory.
            policy_checkpoints: Policy checkpoint paths if required.
            dynamics_checkpoint: Dynamics checkpoint path if required.
        """
        super().__init__(planner_config, "planner", planners)

        if policy_checkpoints is None:
            policy_checkpoints = [None] * len(self.config["agent_configs"])

        self._agent_factories = [
            agents.AgentFactory(
                agent_config=agent_config,
                env_factory=agent_env_factory,
                checkpoint=ckpt,
            )
            for agent_config, agent_env_factory, ckpt in zip(
                self.config["agent_configs"],
                env_factory.env_factories,
                policy_checkpoints,
            )
        ]

        self._dynamics_factory = dynamics.DynamicsFactory(
            dynamics_config=self.config["dynamics_config"],
            env_factory=env_factory,
            checkpoint=dynamics_checkpoint,
        )

        # Preemptively initialize env.
        if issubclass(self.cls, planners.RandomShootingPlanner):
            self.kwargs["env"] = env_factory.get_instance()

    def __call__(self, *args, **kwargs) -> planners.Planner:
        """Creates a Planner instance.

        *args and **kwargs are transferred directly to the Planner constructor.
        PlannerFactory automatically handles the policies, dynamics, env, and
        device arguments.
        """
        device = kwargs.get("device", self.kwargs.get("device", "auto"))

        policies = [
            agent_factory(device=device) for agent_factory in self._agent_factories
        ]
        dynamics_model = self._dynamics_factory(policies=policies, device=device)

        kwargs["policies"] = policies
        kwargs["dynamics"] = dynamics_model
        kwargs["device"] = device

        return super().__call__(*args, **kwargs)


def load(
    planner_config: Union[str, pathlib.Path, Dict[str, Any]],
    env_factory: envs.EnvFactory,
    policy_checkpoints: Optional[Sequence[Optional[str]]] = None,
    dynamics_checkpoint: Optional[str] = None,
    device: str = "auto",
) -> planners.Planner:
    """Loads the planner from a planner_config.

    Args:
        planner_config: Planner config path or dict.
        env_factory: Env factory.
        policy_checkpoints: Policy checkpoint paths if required.
        dynamics_checkpoint: Dynamics checkpoint path if required.
        device: Torch device.

    Returns:
        Planner instance.
    """
    planner_factory = PlannerFactory(
        planner_config=planner_config,
        env_factory=env_factory,
        policy_checkpoints=policy_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
    )
    return planner_factory(device=device)


def evaluate_trajectory(
    value_fns: Iterable[torch.nn.Module],
    decode_fns: Iterable[torch.nn.Module],
    states: torch.Tensor,
    actions: torch.Tensor,
    p_transitions: torch.Tensor,
    q_value: bool = True,
) -> torch.Tensor:
    """Evaluates probability of success for the given trajectory.

    Args:
        value_fns: List of T value functions.
        decoders: List of T decoders.
        states: [T + 1, batch_dims, state_dims] trajectory states.
        actions: [T, batch_dims, state_dims] trajectory actions.
        p_transitions: [T, batch_dims] transition probabilities.
        q_value: Whether to use state-action values (True) or state values (False).

    Returns:
        [batch_dims] Trajectory success probabilities.
    """
    # Compute step success probabilities.
    p_successes = torch.zeros_like(p_transitions)
    if q_value:
        for t, (value_fn, decode_fn) in enumerate(zip(value_fns, decode_fns)):
            state_t = decode_fn(states[t])
            dim_action = torch.sum(~torch.isnan(actions[t, 0])).cpu().item()
            p_successes[t] = value_fn.predict(state_t, actions[t, :, :dim_action])
    else:
        for t, value_fn in enumerate(value_fns):
            state_t = decode_fn(states[t])
            p_successes[t] = value_fn.predict(states[t])
    p_successes = torch.clip(p_successes, min=0, max=1)

    # Discard last transition from T-1 to T, since s_T isn't used.
    p_transitions = p_transitions[:-1]

    # Combine probabilities.
    log_p_success = torch.log(p_successes).sum(dim=0)
    log_p_success += torch.log(p_transitions).sum(dim=0)

    return torch.exp(log_p_success)


def evaluate_plan(
    env: envs.Env,
    action_skeleton: Sequence[Tuple[int, Any]],
    state: np.ndarray,
    actions: np.ndarray,
    gif_path: Optional[Union[str, pathlib.Path]] = None,
) -> np.ndarray:
    """Evaluates the given plan.

    Args:
        env: Sequential env.
        action_skeleton: List of (idx_policy, policy_args) 2-tuples.
        state: Initial state.
        actions: Planned actions [T, A].
        gif_path: Optional path to save a rendered gif.

    Returns:
        Rewards received at each timestep.
    """
    env.set_state(state)

    if gif_path is not None:
        env.record_start()

    rewards = np.zeros(len(action_skeleton), dtype=np.float32)
    for t, (idx_policy, policy_args) in enumerate(action_skeleton):
        action = actions[t, : env.envs[idx_policy].action_space.shape[0]]
        _, reward, _, _ = env.step((action, idx_policy, policy_args))
        rewards[t] = reward

    if gif_path is not None:
        env.record_save(gif_path, stop=True)

    return rewards

    # eval_planner, _ = planners.load(
    #     planner_config={
    #         "planner": "ShootingPlanner",
    #         "planner_kwargs": {"num_samples": 1},
    #         "env": "pybox2d.Sequential2D",
    #         "dynamics": "OracleDynamics",
    #     },
    #     policy_checkpoints=args.policy_checkpoints,
    #     dynamics_checkpoint=args.dynamics_checkpoint,
    #     device=args.device,
    # )
    # assert isinstance(eval_planner, planners.ShootingPlanner)
    # assert isinstance(eval_planner.dynamics, dynamics.OracleDynamics)
    # eval_dynamics: dynamics.OracleDynamics = eval_planner.dynamics
    #
    # # Evaluate.
    # policies = [
    #     agents.ConstantAgent(
    #         action,
    #         env.envs[idx_policy],
    #         # policy.state_space,
    #         policy.action_space,
    #         # policy.observation_space,
    #         device=device,
    #     )
    #     for action, policy, (idx_policy, _) in zip(actions, planner.policies, action_skeleton)
    # ]
    # env.reset()
    # env.set_state(state)
    # eval_dynamics._env = env
    # eval_dynamics._policies = policies
    # eval_planner._policies = policies
    # eval_planner._eval_policies = policies
    # eval_actions, success = eval_planner.plan(observation, action_skeleton)
