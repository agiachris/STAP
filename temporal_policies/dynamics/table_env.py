from dataclasses import field
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import gym
import numpy as np
from temporal_policies.envs.pybullet.table.objects import Rack
from temporal_policies.envs.pybullet.table.primitives import ACTION_CONSTRAINTS
from temporal_policies.envs.pybullet.table_env import TableEnv
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
        self._plan_mode = False

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

        self._observation_space = observation_space
        flat_observation_space = gym.spaces.Box(
            low=observation_space.low.flatten(),
            high=observation_space.high.flatten(),
        )
        self._flat_state_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=flat_observation_space.shape,
            dtype=flat_observation_space.dtype,  # type: ignore
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
            state_space=self.state_space,
            action_space=None,
            checkpoint=checkpoint,
            device=device,
        )

    @property
    def env(self) -> Optional[envs.pybullet.TableEnv]:
        return self._env

    @property
    def state_space(self) -> gym.spaces.Box:
        if self._plan_mode:
            return self._observation_space
        else:
            return self._flat_state_space

    @property
    def flat_state_space(self) -> gym.spaces.Box:
        return self._flat_state_space

    def to(self, device: Union[str, torch.device]) -> LatentDynamics:
        """Transfers networks to device."""
        super().to(device)
        self._observation_mid = self._observation_mid.to(self.device)
        self._observation_range = self._observation_range.to(self.device)
        return self

    def train_mode(self) -> None:
        """Switches to train mode."""
        super().train_mode()
        self._plan_mode = False

    def eval_mode(self) -> None:
        """Switches to eval mode."""
        super().eval_mode()
        self._plan_mode = False

    def plan_mode(self) -> None:
        """Switches to plan mode."""
        super().eval_mode()
        self._plan_mode = True

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Dict[str, List[int]]]],
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
        if self._plan_mode:
            # Return full observation.
            return observation

        assert policy_args is not None
        observation = networks.encoders.TableEnvEncoder.rearrange_observation(
            observation, policy_args, randomize=False
        )

        dynamics_state = self._normalize_state(observation)

        return dynamics_state

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Scale to [-0.5, 0.5].
        state = (state - self._observation_mid) / self._observation_range

        # Flatten state.
        if state.ndim > len(self.state_space.shape):
            state = state.reshape(-1, *self.flat_state_space.shape)
        else:
            state = state.reshape(*self.flat_state_space.shape)

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
        return self.policies[primitive.idx_policy].encoder.encode(
            state, policy_args=primitive.get_policy_args()
        )

    def _apply_handcrafted_dynamics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        predicted_next_state: torch.Tensor,
        primitive: envs.Primitive,
        policy_args: Optional[Dict[str, List[int]]],
    ) -> torch.Tensor:
        """Applies handcrafted dynamics to the state.

        Args:
            state: Current state.
            action: Policy action.
            predicted_next_state: Predicted next state (by network)
            primitive: Current primitive.

        Returns:
            Prediction of next state.
        """
        new_predicted_next_state = predicted_next_state
        primitive_str = str(primitive).lower()
        if "pick" in primitive_str:
            Z_IDX = 2
            new_predicted_next_state = new_predicted_next_state.clone()
            target_object_idx = policy_args["observation_indices"][1]

            new_predicted_next_state[
                ..., target_object_idx, Z_IDX
            ] = ACTION_CONSTRAINTS["max_lift_height"]
            # TODO(klin) the following moves the EE to an awkward position;
            # may need to do quaternion computation for accurate x-y positions
            if "box" in primitive_str:
                new_predicted_next_state[
                    ..., TableEnv.EE_OBSERVATION_IDX, Z_IDX
                ] = ACTION_CONSTRAINTS["max_lift_height"]
                target_object_original_state = state[..., target_object_idx, :]
                new_predicted_next_state[
                    ..., TableEnv.EE_OBSERVATION_IDX, :Z_IDX
                ] = target_object_original_state[..., :Z_IDX]
                new_predicted_next_state[
                    ..., target_object_idx, :Z_IDX
                ] = target_object_original_state[..., :Z_IDX]
        if "place" in primitive_str:
            SRC_OBJ_IDX = 1
            DEST_OBJ_IDX = 2
            new_predicted_next_state = new_predicted_next_state.clone()
            source_object_idx = policy_args["observation_indices"][SRC_OBJ_IDX]
            destination_object_idx = policy_args["observation_indices"][DEST_OBJ_IDX]
            destination_object_state = state[..., destination_object_idx, :]

            if "table" in primitive_str:
                destination_object_surface_offset = 0
            elif "rack" in primitive_str:
                destination_object_surface_offset = Rack.TOP_THICKNESS
            else:
                return new_predicted_next_state

            # hardcoded object heights
            if "box" in primitive_str:
                median_object_height = 0.08
            elif "hook" in primitive_str:
                median_object_height = 0.04
            else:
                return new_predicted_next_state

            new_predicted_next_state[..., source_object_idx, 2] = (
                destination_object_state[..., 2]
                + destination_object_surface_offset
                + median_object_height / 2
            )
        return new_predicted_next_state

    def forward_eval(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        primitive: envs.Primitive,
        use_handcrafted_dynamics_primitives: Optional[List[str]] = None,
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
            use_handcrafted_dynamics_primitives: List of primitives for
                which to use handcrafted dynamics.

        Returns:
            Prediction of next state.
        """
        env_state = state

        # Env state -> dynamics state.
        policy_args = primitive.get_policy_args()
        assert policy_args is not None
        idx_args = policy_args["observation_indices"]
        dynamics_state = self._normalize_state(env_state[..., idx_args, :])

        # Dynamics state -> dynamics state.
        next_dynamics_state = self.forward(
            dynamics_state, action, primitive.idx_policy, policy_args
        )
        next_dynamics_state = next_dynamics_state.clamp(-0.5, 0.5)

        # Update env state with new unnormalized observation.
        next_env_state = env_state.clone()
        next_env_state[..., idx_args, :] = self._unnormalize_state(next_dynamics_state)
        if use_handcrafted_dynamics_primitives is None:
            use_handcrafted_dynamics_primitives = ["pick", "place"]
        for primitive_name in use_handcrafted_dynamics_primitives:
            if primitive_name in str(primitive).lower():
                next_env_state = self._apply_handcrafted_dynamics(
                    env_state, action, next_env_state, primitive, policy_args
                )
                break

        # set states of non existent objects to 0
        non_existent_obj_start_idx = policy_args["shuffle_range"][1]
        next_env_state[..., non_existent_obj_start_idx:, :] = 0
        return next_env_state
