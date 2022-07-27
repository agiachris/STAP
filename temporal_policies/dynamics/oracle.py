from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from temporal_policies import agents, envs
from temporal_policies.dynamics import base as dynamics
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
        self._cache: Dict[Tuple[Tuple, int, Optional[Any]], np.ndarray] = {}

    @property
    def env(self) -> envs.Env:
        """Oracle environment."""
        return self._env

    @env.setter
    def env(self, env: envs.Env) -> None:
        """Updates the oracle environment."""
        self._env = env

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
        prev_state = self.env.get_state()
        prev_primitive = self.env.get_primitive()

        states, actions, p_transitions = super().rollout(
            observation=observation,
            action_skeleton=action_skeleton,
            policies=policies,
            batch_size=batch_size,
            time_index=time_index,
        )

        # Restore env to the way it was before.
        self.env.set_state(prev_state)
        self.env.set_primitive(prev_primitive)

        return states, actions, p_transitions

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
        # print(
        #     f"OracleDynamics.forward(state={state}, action={action}, idx_policy={idx_policy}, policy_args={policy_args})"
        # )
        self.env.set_primitive(idx_policy=idx_policy, policy_args=policy_args)
        self.env.set_state(state)
        action = action[: self.policies[idx_policy].action_space.shape[0]]

        if self._debug:
            self.env.record_start(state)

        self.env.step(action)
        next_state = self.env.get_state()

        if self._debug:
            self.env.record_stop(next_state)

        return next_state

    @tensors.torch_wrap
    def encode(
        self,
        observation: np.ndarray,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> np.ndarray:
        """Returns the current environment state.

        WARNING: This ignores the input observation and instead returns the
        environment's current ground truth state. Be careful that the state
        matches the observation as expected.
        """
        # print(
        #     f"OracleDynamics.encode(observation={observation}, idx_policy={idx_policy}, policy_args={policy_args})"
        # )
        prev_primitive = self.env.get_primitive()

        self.env.set_primitive(idx_policy=idx_policy, policy_args=policy_args)
        env_observation = self.env.get_observation()
        assert observation.ndim == env_observation.ndim
        if (observation != env_observation).any():
            # May happen if self.env is not updated by the dynamics factory.
            raise ValueError("Observation does not match the current env state")
        state = self.env.get_state()

        # Restore env to the way it was before.
        self.env.set_primitive(prev_primitive)

        # Save in cache.
        self._cache[OracleDynamics._get_cache_key(state, idx_policy, policy_args)] = observation

        return state

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def decode(
        self,
        state: np.ndarray,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> np.ndarray:
        """Returns the encoded state for the policy.

        Args:
            state: Current state.
            idx_policy: Index of the executed policy.
            policy_args: Auxiliary policy arguments.

        Returs:
            Encoded state for policy.
        """
        # print(
        #     f"OracleDynamics.decode(state={state}, idx_policy={idx_policy}, policy_args={policy_args})"
        # )
        try:
            observation = self._cache[OracleDynamics._get_cache_key(state, idx_policy, policy_args)]
            reset_env = False
        except KeyError:
            prev_state = self.env.get_state()
            prev_primitive = self.env.get_primitive()

            self.env.set_primitive(idx_policy=idx_policy, policy_args=policy_args)
            self.env.set_state(state)
            observation = self.env.get_observation()

            reset_env = True
            self._cache[OracleDynamics._get_cache_key(state, idx_policy, policy_args)] = observation

        with torch.no_grad():
            t_observation = torch.from_numpy(observation).to(self.device)
            t_policy_state = self.policies[idx_policy].encoder.encode(t_observation)
            policy_state = t_policy_state.cpu().numpy()

        if reset_env:
            # Restore env to the way it was before.
            self.env.set_state(prev_state)
            self.env.set_primitive(prev_primitive)

        return policy_state

    @staticmethod
    def _get_cache_key(state: np.ndarray, idx_policy: int, policy_args: Optional[Any]) -> Tuple[Tuple, int, Any]:
        hash_args = tuple(policy_args) if isinstance(policy_args, list) else policy_args
        return tuple(state), idx_policy, hash_args
