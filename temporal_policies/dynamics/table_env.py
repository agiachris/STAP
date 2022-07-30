import pathlib
from typing import Any, Dict, Optional, Sequence, Type, Union

import numpy as np
import torch

from temporal_policies import agents, envs, networks
from temporal_policies.dynamics.latent import LatentDynamics
from temporal_policies.utils import spaces


class TableEnvDynamics(LatentDynamics):
    """Dynamics model per action with shared latent states.

    We train A dynamics models T_a of the form:

        z^(t+1) = z^(t) + T_a(z^(t), a^(t))

    for every action a.
    """

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[networks.dynamics.PolicyDynamics]],
        network_kwargs: Dict[str, Any],
        env: Optional[envs.pybullet.TableEnv],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Backend network for decoupled dynamics network.
            network_kwargs: Kwargs for network class.
            env: TableEnv required for planning (not training).
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
        if self.env is None:
            # Placeholder - state space not needed for training.
            state_space = policies[0].state_space
        else:
            # State space required for planning.
            state_space = self.env.full_observation_space

        super().__init__(
            policies=policies,
            network_class=parent_network_class,
            network_kwargs=parent_network_kwargs,
            state_space=state_space,
            action_space=None,
            checkpoint=checkpoint,
            device=device,
        )

    @property
    def env(self) -> Optional[envs.pybullet.TableEnv]:
        return self._env

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Encodes the observation into a dynamics state.

        During training, the dynamics state is equivalent to the policy state
        (normalized vector containing state for 3 objects). During planning, the
        dynamics state is equivalent to the environment observation
        (unnormalized matrix containing state for all objects).

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded latent state vector.
        """
        if isinstance(idx_policy, int):
            policy = self.policies[idx_policy]
            if tuple(observation.shape[1:]) != policy.state_space.shape:
                # Full TableEnv observation.
                return observation

            # [B, O] => [B, Z].
            return policy.encoder.encode(observation)

        # Assume all encoders are the same.
        return self.policies[0].encoder.encode(observation)

    def decode(
        self,
        state: torch.Tensor,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Decodes the dynamics state into policy states.

        This is only used during planning, not training, so the input state will
        be the environment state.

        Args:
            state: Full TableEnv observation.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Decoded observation.
        """
        assert self.env is not None
        env_state = state

        idx_args = self.env.get_arg_indices(idx_policy, policy_args)
        observation = env_state[..., idx_args, :].view(*state.shape[:-2], -1)
        policy_state = self.encode(observation, idx_policy, policy_args)

        return policy_state

    def forward_eval(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Predicts the next state for planning.

        During planning, the state is an unnormalized matrix with one row for
        each object. This gets transformed into a normalized policy state vector
        according to the current primitive and fed to the dynamics model. The
        row entries in the state corresponding to the objects involved with the
        primitive are updated according to the dynamics prediction.

        Args:
            state: Current state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next state.
        """
        assert self.env is not None
        env_state = state

        # Env state -> dynamics state (= policy state).
        dynamics_state = self.decode(env_state, idx_policy, policy_args)

        # Dynamics state -> dynamics state.
        next_dynamics_state = self.forward(
            dynamics_state, action, idx_policy, policy_args
        )

        # Dynamics state -> new unnormalized observation.
        next_dynamics_state = torch.from_numpy(
            spaces.transform(
                np.clip(
                    next_dynamics_state.cpu().numpy(),
                    self.policies[idx_policy].state_space.low,
                    self.policies[idx_policy].state_space.high,
                ),
                self.policies[idx_policy].state_space,
                self.env.observation_space,
            )
        ).to(next_dynamics_state.device)

        # Update env state with new unnormalized observation.
        next_env_state = env_state.clone()
        idx_args = self.env.get_arg_indices(idx_policy, policy_args)
        next_env_state[..., idx_args, :] = next_dynamics_state.view(
            *next_dynamics_state.shape[:-1], len(idx_args), -1
        )

        return next_env_state
