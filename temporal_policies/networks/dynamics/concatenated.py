from typing import Any, Dict, Type

import torch  # type: ignore


class ConcatenatedDynamics(torch.nn.Module):
    """Dynamics model `T_a` for a single policy `a` constructed from
    concatenated latent spaces of all policies."""

    def __init__(
        self,
        num_policies: int,
        network_class: Type[torch.nn.Module],
        network_kwargs: Dict[str, Any] = {},
    ):
        """Constructs `num_networks` instances of the given backend network,
        whose results will be concatenated for the output latent predictions.

        Args:
            num_policies: Number of policies.
            network_class: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__()
        self._num_policies = num_policies
        self.models = torch.nn.ModuleList(
            [network_class(**network_kwargs) for _ in range(num_policies)]
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Calls all subnetworks and concatenates the results.

        Computes z' = T_a(z, theta_a).

        Args:
            latent: Current latent state.
            action: Policy action (for a single policy).

        Returns:
            Concatenated latent prediction z' for a single policy action.
        """
        policy_latents = torch.reshape(
            latent,
            (
                *latent.shape[:-1],
                self._num_policies,
                latent.shape[-1] // self._num_policies,
            ),
        )
        next_latents = [
            model_a(policy_latents[..., i, :], action)
            for i, model_a in enumerate(self.models)
        ]
        next_latents = torch.cat(next_latents, dim=-1)
        return next_latents
