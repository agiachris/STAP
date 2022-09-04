import abc
from typing import Any, Dict, Optional, Sequence, Type, Union

import gym
import torch

from temporal_policies import networks
from temporal_policies.utils import configs


class Dynamics(torch.nn.Module):
    """Dynamics model s' = T(s, a, idx_policy)."""

    def __init__(
        self,
        policies: Sequence,
        network_class: Union[str, Type["PolicyDynamics"]],
        network_kwargs: Dict[str, Any],
        state_spaces: Optional[Sequence[gym.spaces.Box]] = None,
    ):
        """Constructs a dynamics network per policy.

        Args:
            policies: Policies.
            network_class: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__()

        if state_spaces is None:
            state_spaces = [policy.state_space for policy in policies]

        network_class = configs.get_class(network_class, networks)
        self.models = torch.nn.ModuleList(
            [
                network_class(
                    state_space=state_space,
                    action_space=policy.action_space,
                    **network_kwargs
                )
                for policy, state_space in zip(policies, state_spaces)
            ]
        )

    def forward(
        self,
        latents: torch.Tensor,
        policy_indices: Union[int, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the next latent state using separate dynamics model per
        action.

        Args:
            latents: Current latent states.
            policy_indices: Indices of executed policy.
            actions: Policy actions.

        Returns:
            Prediction of next latent state.
        """
        if isinstance(policy_indices, torch.Tensor):
            next_latents = torch.full_like(latents, float("nan"))
            for i, policy_model in enumerate(self.models):
                idx_policy = policy_indices == i
                next_latents[idx_policy] = policy_model(
                    latents[idx_policy], actions[idx_policy]
                )
        else:
            next_latents = self.models[policy_indices](latents, actions)

        return next_latents


class PolicyDynamics(torch.nn.Module, abc.ABC):
    """Primitive dynamics model s' = T_{idx_policy}(s, a)."""

    def __init__(self, state_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the next latent state.

        Args:
            latent: Current latent state.
            action: Policy action.

        Returns:
            Prediction of next latent state.
        """
        pass
