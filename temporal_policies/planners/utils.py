import pathlib
from typing import Any, Dict, Optional, Iterable, Sequence, Tuple, Union

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies import agents, dynamics, envs, planners
from temporal_policies.utils import configs, tensors


class PlannerFactory(configs.Factory):
    """Planner factory."""

    def __init__(
        self,
        config: Union[str, pathlib.Path, Dict[str, Any]],
        env: envs.SequentialEnv,
        policy_checkpoints: Optional[
            Sequence[Optional[Union[str, pathlib.Path]]]
        ] = None,
        dynamics_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Creates the planner factory from a planner_config.

        Args:
            config: Planner config path or dict.
            env: Sequential env.
            policy_checkpoints: Policy checkpoint paths if required.
            dynamics_checkpoint: Dynamics checkpoint path if required.
            device: Torch device.
        """
        super().__init__(config, "planner", planners)

        # TODO: Get agent_configs from the checkpoint.
        if policy_checkpoints is None:
            policy_checkpoints = [None] * len(self.config["agent_configs"])

        policies = [
            agents.load(
                config=agent_config,
                env=policy_env,
                checkpoint=ckpt,
            )
            for agent_config, policy_env, ckpt in zip(
                self.config["agent_configs"],
                env.envs,
                policy_checkpoints,
            )
        ]

        dynamics_model = dynamics.load(
            config=self.config["dynamics_config"],
            checkpoint=dynamics_checkpoint,
            policies=policies,
            env=env,
            device=device,
        )

        self.kwargs["policies"] = policies
        self.kwargs["dynamics"] = dynamics_model
        self.kwargs["device"] = device


def load(
    config: Union[str, pathlib.Path, Dict[str, Any]],
    env: envs.SequentialEnv,
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]] = None,
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    device: str = "auto",
    **kwargs,
) -> planners.Planner:
    """Loads the planner from config.

    Args:
        config: Planner config path or dict.
        env: Sequential env.
        policy_checkpoints: Policy checkpoint paths if required.
        dynamics_checkpoint: Dynamics checkpoint path if required.
        device: Torch device.
        **kwargs: Planner constructor kwargs.

    Returns:
        Planner instance.
    """
    planner_factory = PlannerFactory(
        config=config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    return planner_factory(**kwargs)


@tensors.batch(dims=1)
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
        states: [batch_dims, T + 1, state_dims] trajectory states.
        actions: [batch_dims, T, state_dims] trajectory actions.
        p_transitions: [batch_dims, T] transition probabilities.
        q_value: Whether to use state-action values (True) or state values (False).

    Returns:
        [batch_dims] Trajectory success probabilities.
    """
    # Compute step success probabilities.
    p_successes = torch.zeros_like(p_transitions)
    if q_value:
        for t, (value_fn, decode_fn) in enumerate(zip(value_fns, decode_fns)):
            policy_state = decode_fn(states[:, t])
            dim_action = torch.sum(~torch.isnan(actions[0, t])).cpu().item()
            action = actions[:, t, :dim_action]
            p_successes[:, t] = value_fn.predict(policy_state, action)
    else:
        for t, value_fn in enumerate(value_fns):
            policy_state = decode_fn(states[:, t])
            p_successes[:, t] = value_fn.predict(policy_state)
    p_successes = torch.clip(p_successes, min=0, max=1)

    # Discard last transition from T-1 to T, since s_T isn't used.
    p_transitions = p_transitions[:, :-1]

    # Combine probabilities.
    log_p_success = torch.log(p_successes).sum(dim=-1)
    log_p_success += torch.log(p_transitions).sum(dim=-1)

    return torch.exp(log_p_success)


def evaluate_plan(
    env: envs.SequentialEnv,
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
