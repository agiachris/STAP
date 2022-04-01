from typing import Any, Dict, Sequence, Type

import torch  # type: ignore


from temporal_policies import agents
from temporal_policies.networks.dynamics import concatenated


class DecoupledDynamics(torch.nn.Module):
    """Network module for the decoupled dynamics model."""

    def __init__(
        self,
        policies: Sequence[agents.Agent],
        network_class: Type[torch.nn.Module],
        network_kwargs: Dict[str, Any],
    ):
        """Constructs `num_networks` instances of ConcatenatedDynamicsModel.

        Args:
            num_policies: Number of policies.
            network_class: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__()

        self._num_policies = len(policies)
        self.models = torch.nn.ModuleList(
            [
                concatenated.ConcatenatedDynamics(
                    self._num_policies,
                    network_class,
                    dict(
                        network_kwargs,
                        action_space=policy.action_space,
                    ),
                )
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
            latents: Current latent state.
            policy_indices: Index of executed policy.
            actions: Policy action.

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
