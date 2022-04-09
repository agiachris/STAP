import abc
from typing import Any, Optional, Sequence, Tuple, Union

import gym  # type: ignore
import torch  # type: ignore

from temporal_policies import agents
from temporal_policies.utils import spaces, tensors


class Dynamics(abc.ABC):
    """Base dynamics class."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        state_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
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
    def state_space(self) -> gym.spaces.Space:
        """State space."""
        return self._state_space

    @property
    def action_space(self) -> gym.spaces.Space:
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
        state: torch.Tensor,
        action_skeleton: Sequence[Tuple[int, Any]],
        policies: Optional[Sequence[agents.Agent]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rolls out trajectories according to the action skeleton.

        Args:
            state: Start state (supports up to one batch dimension for now).
            action_skeleton: List of (idx_policy, policy_args) 2-tuples.
            policies: Optional policies to use. Otherwise uses `self.policies`.

        Returns:
            3-tuple (
                states [T + 1, batch_size],
                actions [T, batch_size],
                p_transitions [T, batch_size],
            ).
        """
        if policies is None:
            policies = [self.policies[idx_policy] for idx_policy, _ in action_skeleton]

        squeeze = state.dim == len(self.state_space.shape)
        if squeeze:
            state = state.unsqueeze(0)

        # Initialize variables.
        batch_size = state.shape[0]
        T = len(action_skeleton)
        states = spaces.null_tensor(
            self.state_space, (T + 1, batch_size), device=self.device
        )
        states[0] = state
        actions = spaces.null_tensor(
            self.action_space, (T, batch_size), device=self.device
        )
        p_transitions = torch.ones(
            (T, batch_size), dtype=torch.float32, device=self.device
        )

        # Rollout.
        for t, (idx_policy, policy_args) in enumerate(action_skeleton):
            policy_state = self.decode(state, idx_policy, policy_args)
            action = policies[t].actor.predict(policy_state)
            actions[t, :, : action.shape[-1]] = action

            state = self.forward(state, idx_policy, action, policy_args)
            states[t + 1] = state

        if squeeze:
            states = states.squeeze(dim=1)
            actions = actions.squeeze(dim=1)
            p_transitions = p_transitions.squeeze(dim=1)

        return states, actions, p_transitions

    @abc.abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        action: torch.Tensor,
        policy_args: Optional[Any] = None,
    ) -> torch.Tensor:
        """Predicts the next state given the current state and action.

        Args:
            state: Current state.
            idx_policy: Index of executed policy.
            action: Policy action.
            policy_args: Auxiliary policy arguments.

        Returns:
            Prediction of next state.
        """
        raise NotImplementedError

    def encode(
        self, observation: Any, idx_policy: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Encodes the observation into a dynamics state.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.

        Returns:
            Encoded observation.
        """

        @tensors.vmap(dims=1)
        def _encode(idx_policy: Union[int, torch.Tensor], observation: Any):
            if isinstance(idx_policy, torch.Tensor):
                idx_policy = idx_policy.item()
            return self.policies[idx_policy].encoder(observation)

        return _encode(idx_policy, observation)

    def decode(
        self,
        state: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Optional[Any] = None,
    ) -> Any:
        """Decodes the dynamics state into policy states.

        Args:
            state: Encoded state state.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Decoded observation.
        """
        return state
