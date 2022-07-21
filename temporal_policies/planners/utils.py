import pathlib
from typing import Any, Dict, Optional, Iterable, List, Sequence, Union

import numpy as np
import torch
import yaml

from temporal_policies import agents, dynamics, envs, planners
from temporal_policies.utils import configs, tensors


class PlannerFactory(configs.Factory):
    """Planner factory."""

    def __init__(
        self,
        config: Union[str, pathlib.Path, Dict[str, Any]],
        env: envs.Env,
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

        def replace_config(config, old: str, new: str):
            config_yaml: str = yaml.dump(config)
            config_yaml = config_yaml.replace(old, new)
            config = yaml.safe_load(config_yaml)
            return config

        super().__init__(config, "planner", planners)

        if policy_checkpoints is None:
            policy_checkpoints = [None] * len(self.config["agent_configs"])
        else:
            # Get agent configs from checkpoints.
            for idx_policy, policy_checkpoint in enumerate(policy_checkpoints):
                if policy_checkpoint is None:
                    continue
                agent_config = str(
                    pathlib.Path(policy_checkpoint).parent / "agent_config.yaml"
                )
                self.config["agent_configs"][idx_policy] = replace_config(
                    self.config["agent_configs"][idx_policy],
                    "{AGENT_CONFIG}",
                    agent_config,
                )

        # Get dynamics config from checkpoint.
        if dynamics_checkpoint is not None:
            dynamics_config = str(
                pathlib.Path(dynamics_checkpoint).parent / "dynamics_config.yaml"
            )
            self.config["dynamics_config"] = replace_config(
                self.config["dynamics_config"], "{DYNAMICS_CONFIG}", dynamics_config
            )

        if isinstance(env, envs.pybox2d.Sequential2D):
            # TODO: Implement policy envs.
            raise NotImplementedError("Need to implement policy envs")
        policies = [
            agents.load(
                config=agent_config,
                # env=policy_env,
                env=env,
                checkpoint=ckpt,
            )
            # for agent_config, policy_env, ckpt in zip(
            #     self.config["agent_configs"],
            #     env.envs,
            #     policy_checkpoints,
            # )
            for agent_config, ckpt in zip(
                self.config["agent_configs"], policy_checkpoints
            )
        ]

        # Make sure all policy checkpoints are not None for dynamics.
        dynamics_policy_checkpoints: Optional[List[Union[str, pathlib.Path]]] = []
        for policy_checkpoint in policy_checkpoints:
            if policy_checkpoint is None:
                dynamics_policy_checkpoints = None
                break
            assert dynamics_policy_checkpoints is not None
            dynamics_policy_checkpoints.append(policy_checkpoint)

        dynamics_model = dynamics.load(
            config=self.config["dynamics_config"],
            checkpoint=dynamics_checkpoint,
            policies=policies,
            policy_checkpoints=dynamics_policy_checkpoints,
            env=env,
            device=device,
        )

        self.kwargs["policies"] = policies
        self.kwargs["dynamics"] = dynamics_model
        self.kwargs["device"] = device


def load(
    config: Union[str, pathlib.Path, Dict[str, Any]],
    env: envs.Env,
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
            dim_action = int(torch.sum(~torch.isnan(actions[0, t])).cpu().item())
            action = actions[:, t, :dim_action]
            p_successes[:, t] = value_fn.predict(policy_state, action)  # type: ignore
    else:
        for t, value_fn in enumerate(value_fns):
            policy_state = decode_fn(states[:, t])
            p_successes[:, t] = value_fn.predict(policy_state)  # type: ignore
    p_successes = torch.clip(p_successes, min=0, max=1)

    # Discard last transition from T-1 to T, since s_T isn't used.
    p_transitions = p_transitions[:, :-1]

    # Combine probabilities.
    log_p_success = torch.log(p_successes).sum(dim=-1)
    log_p_success += torch.log(p_transitions).sum(dim=-1)

    return torch.exp(log_p_success)


def evaluate_plan(
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
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
    for t, primitive in enumerate(action_skeleton):
        env.set_primitive(primitive)
        action = actions[t, : env.action_space.shape[0]]
        _, reward, _, _ = env.step(action)
        rewards[t] = reward

    if gif_path is not None:
        env.record_stop()
        env.record_save(gif_path, reset=True)

    return rewards
