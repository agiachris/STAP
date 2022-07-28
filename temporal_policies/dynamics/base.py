import abc
from typing import Any, Optional, Sequence, Tuple, Union

import gym
import torch

from temporal_policies import agents, envs
from temporal_policies.utils import spaces, tensors


class Dynamics(abc.ABC):
    """Base dynamics class."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        state_space: Optional[gym.spaces.Box] = None,
        action_space: Optional[gym.spaces.Box] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            state_space: Optional state space. Default set to the first policy
                state space.
            action_space: Optional action space. Default set as an overlay of
                all policy action spaces.
        """
        self._policies = policies

        if state_space is None:
            if not all(
                policy.state_space == policies[0].state_space for policy in policies[1:]
            ):
                raise ValueError("All policy state spaces must be the same.")
            self._state_space = policies[0].state_space
        else:
            self._state_space = state_space

        if action_space is None:
            if not all(
                isinstance(policy.action_space, gym.spaces.Box) for policy in policies
            ):
                raise ValueError("All policy action spaces must be boxes.")
            self._action_space = spaces.overlay_boxes(
                [policy.action_space for policy in policies]
            )
        else:
            self._action_space = action_space

        self.to(device)

    @property
    def policies(self) -> Sequence[agents.Agent]:
        """Ordered list of policies used to perform the task."""
        return self._policies

    @property
    def state_space(self) -> gym.spaces.Box:
        """State space."""
        return self._state_space

    @property
    def action_space(self) -> gym.spaces.Box:
        """Action space."""
        return self._action_space

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Dynamics":
        """Transfers networks to device."""
        self._device = torch.device(tensors.device(device))
        for policy in self.policies:
            policy.to(self.device)
        return self

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

        state = self.encode(
            observation, action_skeleton[0].idx_policy, action_skeleton[0].policy_args
        )
        _batch_size = 1 if batch_size is None else batch_size
        state = state.unsqueeze(0).repeat(_batch_size, *([1] * len(state.shape)))

        # Initialize variables.
        T = len(action_skeleton)
        states = spaces.null_tensor(
            self.state_space, (_batch_size, T + 1), device=self.device
        )
        states[:, 0] = state
        actions = spaces.null_tensor(
            self.action_space, (_batch_size, T), device=self.device
        )
        p_transitions = torch.ones(
            (_batch_size, T), dtype=torch.float32, device=self.device
        )

        # Rollout.
        for t, primitive in enumerate(action_skeleton):
            # Dynamics state -> policy state.
            policy_state = self.decode(
                state, primitive.idx_policy, primitive.policy_args
            )
            policy = policies[t] if time_index else policies[primitive.idx_policy]
            action = policy.actor.predict(policy_state)
            actions[:, t, : action.shape[-1]] = action

            # Dynamics state -> dynamics state.
            state = self.forward_eval(
                state, action, primitive.idx_policy, primitive.policy_args
            )
            states[:, t + 1] = state

        if batch_size is None:
            return states[0], actions[0], p_transitions[0]

        return states, actions, p_transitions

    @abc.abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Predicts the next state given the current state and action.

        Args:
            state: Current state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next state.
        """
        raise NotImplementedError

    def forward_eval(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Predicts the next state for planning.

        Args:
            state: Current state.
            action: Policy action.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next state.
        """
        return self.forward(state, action, idx_policy, policy_args)

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Encodes the observation into a dynamics state.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded observation.
        """

        @tensors.vmap(dims=1)
        def _encode(
            t_idx_policy: Union[int, torch.Tensor], observation: Any
        ) -> torch.Tensor:
            if isinstance(t_idx_policy, torch.Tensor):
                idx_policy = int(t_idx_policy.item())
            else:
                idx_policy = t_idx_policy
            return self.policies[idx_policy].encoder.encode(observation)

        return _encode(idx_policy, observation)

    def decode(
        self,
        state: torch.Tensor,
        idx_policy: int,
        policy_args: Optional[Any],
    ) -> torch.Tensor:
        """Decodes the dynamics state into policy states.

        This is only used during planning, not training.

        Args:
            state: Encoded state state.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Decoded observation.
        """
        return state
