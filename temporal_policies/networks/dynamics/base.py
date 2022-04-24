import abc
from typing import Any, Dict, Sequence, Type, Union

import torch  # type: ignore

from temporal_policies import networks
from temporal_policies.utils import configs


class Dynamics(torch.nn.Module):
    """Dynamics model s' = T(s, a, idx_policy)."""

    def __init__(
        self,
        policies: Sequence,
        network_class: Union[str, Type["PolicyDynamics"]],
        network_kwargs: Dict[str, Any],
    ):
        """Constructs a dynamics network per policy.

        Args:
            policies: Policies.
            network_class: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__()

        network_class = configs.get_class(network_class, networks)
        self.models = torch.nn.ModuleList(
            [
                network_class(action_space=policy.action_space, **network_kwargs)
                for policy in policies
            ]
        )

    def forward(
        self,
        latents: torch.Tensor,
        policy_indices: torch.Tensor,
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
        next_latents = torch.full_like(latents, float("nan"))
        for i, policy_model in enumerate(self.models):
            idx_policy = policy_indices == i
            next_latents[idx_policy] = policy_model(
                latents[idx_policy], actions[idx_policy]
            )
        return next_latents


class PolicyDynamics(torch.nn.Module, abc.ABC):
    """Primitive dynamics model s' = T_{idx_policy}(s, a)."""

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
