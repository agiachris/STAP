import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import torch

from temporal_policies import agents, envs, networks
from temporal_policies.dynamics.latent import LatentDynamics
from temporal_policies.utils import spaces


class TableEnvDynamics(LatentDynamics):
    """Dynamics model per action with shared latent states.

    We train A dynamics models T_a of the form:

        z^(t+1) = z^(t) + T_a(z^(t), a^(t))

    for every action a..

    policy observation: np.ndarray
    env observation: Dict[str, np.ndarray]

    dynamics state: np.ndarray
    next dynamics state: np.ndarray

    STANDARD
    encode: env -> dynamics
    forward: dynamics -> dynamics
    rollout: dynamics, actions, p_transitions
    decode: dynamics -> policy

    NEW
    encode: env -> dynamics
    forward: dynamics -> dynamics
    decode_observation: env, dynamics -> env
    rollout: env, actions, p_transitions
    decode: env -> policy

    """

    def __init__(
        self,
        env: envs.pybullet.TableEnv,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[networks.dynamics.PolicyDynamics]],
        network_kwargs: Dict[str, Any],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Backend network for decoupled dynamics network.
            network_kwargs: Kwargs for network class.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        parent_network_class = networks.dynamics.Dynamics
        parent_network_kwargs = {
            "policies": policies,
            "network_class": network_class,
            "network_kwargs": network_kwargs,
        }
        self._env = env

        super().__init__(
            policies=policies,
            network_class=parent_network_class,
            network_kwargs=parent_network_kwargs,
            state_space=self.env.full_observation_space,
            action_space=None,
            checkpoint=checkpoint,
            device=device,
        )

    @property
    def env(self) -> envs.pybullet.TableEnv:
        return self._env

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Returns the full TableEnv observation as is.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded latent state vector.
        """
        if isinstance(idx_policy, int):
            # [B, O] => [B, Z].
            return self.policies[idx_policy].encoder.encode(observation)

        # Assume all encoders are the same.
        return self.policies[0].encoder.encode(observation)

    def decode(
        self,
        state: torch.Tensor,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Decodes the dynamics state into policy states.

        Args:
            state: Encoded state state.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Decoded observation.
        """
        idx_args = self.env.get_arg_indices(idx_policy, policy_args)
        policy_state = state[..., idx_args, :].view(*state.shape[:-2], -1)
        return policy_state

    def update_state(
        self,
        state: torch.Tensor,
        dynamics_state: torch.Tensor,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        idx_args = self.env.get_arg_indices(idx_policy, policy_args)
        next_state = state.clone()
        next_state[..., idx_args, :] = dynamics_state.view(
            *dynamics_state.shape[:-1], len(idx_args), -1
        )
        return next_state

    def rollout(
        self,
        observation: torch.Tensor,
        action_skeleton: Sequence[envs.Primitive],
        policies: Optional[Sequence[agents.Agent]] = None,
        batch_size: Optional[int] = None,
        time_index: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rolls out trajectories according to the action skeleton.

        Args:
            observation: Initial observation.
            action_skeleton: List of primitives.
            policies: Optional policies to use. Otherwise uses `self.policies`.
            batch_size: Number of trajectories to roll out.
            time_index: True if policies are indexed by time instead of idx_policy.

        Returns:
            3-tuple (
                states [batch_size, T + 1],
                actions [batch_size, T],
                p_transitions [batch_size, T],
            ).
        """
        if policies is None:
            policies = [
                self.policies[primitive.idx_policy] for primitive in action_skeleton
            ]
            time_index = False

        _batch_size = 1 if batch_size is None else batch_size
        env_state = observation.unsqueeze(0).repeat(
            _batch_size, *([1] * len(observation.shape))
        )

        # Initialize variables.
        T = len(action_skeleton)
        states = spaces.null_tensor(
            self.state_space, (_batch_size, T + 1), device=self.device
        )
        states[:, 0] = env_state
        actions = spaces.null_tensor(
            self.action_space, (_batch_size, T), device=self.device
        )
        p_transitions = torch.ones(
            (_batch_size, T), dtype=torch.float32, device=self.device
        )

        # Rollout.
        for t, primitive in enumerate(action_skeleton):
            # Env state -> policy state.
            policy_state = self.decode(
                env_state, primitive.idx_policy, primitive.policy_args
            )
            policy = policies[t] if time_index else policies[primitive.idx_policy]
            action = policy.actor.predict(policy_state)
            actions[:, t, : action.shape[-1]] = action

            # Env state -> dynamics state.
            dynamics_state = policy_state

            # Dynamics state -> dynamics state.
            dynamics_state = self.forward(
                dynamics_state, action, primitive.idx_policy, primitive.policy_args
            )

            # Dynamics state -> env state.
            env_state = self.update_state(
                env_state, dynamics_state, primitive.idx_policy, primitive.policy_args
            )

            states[:, t + 1] = env_state

        if batch_size is None:
            return states[0], actions[0], p_transitions[0]

        return states, actions, p_transitions
