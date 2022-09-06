import pathlib
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import gym
import numpy as np
import torch

from temporal_policies import agents, envs, networks
from temporal_policies.dynamics.latent import LatentDynamics


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
        self._env = env
        self.planning = self.env is not None

        if self.env is None:
            observation_space = policies[0].observation_space
        else:
            observation_space = self.env.observation_space

        self._observation_mid = torch.from_numpy(
            (observation_space.low[0] + observation_space.high[0]) / 2
        )
        self._observation_range = torch.from_numpy(
            observation_space.high[0] - observation_space.low[0]
        )

        self._flat_state_space = gym.spaces.Box(
            low=observation_space.low.flatten(),
            high=observation_space.high.flatten(),
        )

        if self.planning:
            state_space = observation_space
        else:
            state_space = gym.spaces.Box(
                low=-0.5,
                high=0.5,
                shape=self.flat_state_space.shape,
                dtype=self.flat_state_space.dtype,  # type: ignore
            )

        parent_network_class = networks.dynamics.Dynamics
        parent_network_kwargs = {
            "policies": policies,
            "network_class": network_class,
            "network_kwargs": network_kwargs,
            "state_spaces": [self.flat_state_space] * len(policies),
        }
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

    @property
    def flat_state_space(self) -> gym.spaces.Box:
        return self._flat_state_space

    def to(self, device: Union[str, torch.device]) -> LatentDynamics:
        """Transfers networks to device."""
        super().to(device)
        self._observation_mid.to(self.device)
        self._observation_range.to(self.device)
        return self

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Dict[str, List[int]]]],
        envs: Optional[Sequence[envs.pybullet.TableEnv]] = None,
    ) -> torch.Tensor:
        """Encodes the observation into a dynamics state.

        During training, the dynamics state is equivalent to the policy state
        (normalized vector containing state for 3 objects) appended with
        additional object states. During planning, the dynamics state is
        equivalent to the environment observation (unnormalized matrix
        containing state for all objects).

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded latent state vector.
        """
        if self.planning:
            # Return full observation.
            return observation

        if isinstance(policy_args, np.ndarray):
            observation_indices = policy_args
        else:
            assert policy_args is not None
            observation_indices = np.array(policy_args["observation_indices"])

        observation = networks.encoders.TableEnvEncoder.rearrange_observation(
            observation, observation_indices
        )

        dynamics_state = self._normalize_state(observation)

        return dynamics_state

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Scale to [-0.5, 0.5].
        state = (state - self._observation_mid) / self._observation_range

        # Flatten state.
        state = state.reshape(-1, *self.flat_state_space.shape)

        return state

    def _unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Unflatten state if planning.
        state = state.reshape(-1, *self.state_space.shape)

        # Scale from [-0.5, 0.5].
        state = state * self._observation_range + self._observation_mid

        return state

    def decode(self, state: torch.Tensor, primitive: envs.Primitive) -> torch.Tensor:
        """Decodes the dynamics state into policy states.

        This is only used during planning, not training, so the input state will
        be the environment state.

        Args:
            state: Full TableEnv observation.
            primitive: Current primitive.

        Returns:
            Decoded observation.
        """
        assert self.env is not None
        self.env.set_primitive(primitive)
        return self.policies[primitive.idx_policy].encoder.encode(
            state, env=self.env, policy_args=primitive.get_policy_args()
        )

    def forward_eval(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        primitive: envs.Primitive,
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

        # Env state -> dynamics state.
        policy_args = primitive.get_policy_args()
        assert policy_args is not None
        idx_args = policy_args["observation_indices"]
        dynamics_state = self._normalize_state(env_state[..., idx_args, :])

        # Dynamics state -> dynamics state.
        next_dynamics_state = self.forward(
            dynamics_state, action, primitive.idx_policy, primitive.get_policy_args()
        )
        next_dynamics_state = next_dynamics_state.clamp(-0.5, 0.5)

        # Update env state with new unnormalized observation.
        next_env_state = env_state.clone()
        next_env_state[..., idx_args, :] = self._unnormalize_state(next_dynamics_state)

        return next_env_state
