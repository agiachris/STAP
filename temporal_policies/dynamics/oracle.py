from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from temporal_policies import agents, envs, networks
from temporal_policies.dynamics import base as dynamics
from temporal_policies.networks.encoders.oracle import OracleDecoder, OracleEncoder
from temporal_policies.utils import tensors


class OracleDynamics(dynamics.Dynamics):
    """Dummy dynamics model that uses simulation to get the next state."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        env: envs.Env,
        debug: bool = False,
        device: str = "auto",
    ):
        """Constructs the oracle dynamics wrapper.

        Args:
            policies: Ordered list of all policies.
            env: Oracle environment.
            device: Torch device.
        """
        super().__init__(policies=policies, state_space=env.state_space, device=device)

        if len(self.state_space.shape) != 1:
            raise NotImplementedError(
                "tensors.vmap(self.forward) assumes state_space is 1-dimensional."
            )

        self._env = env
        self._debug = debug

        self._oracle_decoder = OracleDecoder(self.env)
        self._oracle_encoder = OracleEncoder(self.env)

        if not hasattr(self.env, "_oracle_sim_result"):
            self.env._oracle_sim_result: Dict[Tuple[str, Tuple, Tuple], float] = {}  # type: ignore

    @property
    def env(self) -> envs.Env:
        """Oracle environment."""
        return self._env

    @env.setter
    def env(self, env: envs.Env) -> None:
        """Updates the oracle environment."""
        self._env = env

    def reset_cache(self) -> None:
        self.env._oracle_sim_result.clear()  # type: ignore
        self._oracle_encoder.reset_cache()
        self._oracle_decoder.reset_cache()

    def rollout(
        self,
        observation: torch.Tensor,
        action_skeleton: Sequence[envs.Primitive],
        policies: Optional[Sequence[agents.Agent]] = None,
        batch_size: Optional[int] = None,
        time_index: bool = False,
        state_requires_grad: bool = False,
        action_requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        prev_state = self.env.get_state()
        prev_primitive = self.env.get_primitive()

        states, actions = super().rollout(
            observation=observation,
            action_skeleton=action_skeleton,
            policies=policies,
            batch_size=batch_size,
            time_index=time_index,
            state_requires_grad=state_requires_grad,
            action_requires_grad=action_requires_grad,
        )

        # Restore env to the way it was before.
        self.env.set_state(prev_state)
        self.env.set_primitive(prev_primitive)

        return states, actions

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def forward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> np.ndarray:
        """Simulates a single step in the environment.

        Args:
            state: Current state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Next state.
        """
        self.env.set_primitive(idx_policy=idx_policy, policy_args=policy_args)
        self.env.set_state(state)
        action = action[: self.policies[idx_policy].action_space.shape[0]]

        if self._debug:
            self.env.record_start(state)

        result = self.env.step(action)
        next_state = self.env.get_state()

        # Cache results for oracle critic.
        self.env._oracle_sim_result[  # type: ignore
            str(self.env.get_primitive()), tuple(state), tuple(action)
        ] = result

        if self._debug:
            self.env.record_stop(next_state)

        return next_state

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Any]],
    ) -> torch.Tensor:
        """Returns the current environment state.

        WARNING: This ignores the input observation and instead returns the
        environment's current ground truth state. Be careful that the state
        matches the observation as expected.
        """
        assert isinstance(idx_policy, int)

        self.env.set_primitive(idx_policy=idx_policy, policy_args=policy_args)
        state = self._oracle_encoder(observation, policy_args)

        return state

    def decode(
        self,
        state: torch.Tensor,
        primitive: envs.Primitive,
    ) -> torch.Tensor:
        """Returns the encoded state for the policy.

        Args:
            state: Current state.
            primitive: Current primitive.

        Returs:
            Encoded state for policy.
        """
        self.env.set_primitive(primitive)

        observation = self._oracle_decoder(state)

        with torch.no_grad():
            policy_encoder = self.policies[primitive.idx_policy].encoder
            if isinstance(policy_encoder.network, networks.encoders.OracleEncoder):
                # Oracle policy state is simply the env state.
                policy_state = state
            else:
                # Encode the policy state.
                policy_state = policy_encoder.encode(
                    observation, primitive.get_policy_args()
                )

        return policy_state
