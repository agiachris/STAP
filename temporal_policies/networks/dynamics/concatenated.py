from typing import Any, Dict, Type, Union

from temporal_policies import networks
from temporal_policies.networks.dynamics.base import PolicyDynamics
from temporal_policies.utils import configs

import gym
import torch


class ConcatenatedDynamics(PolicyDynamics):
    """Dynamics model for the action space of a single policy, constructed from
    concatenated latent spaces of all policies."""

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        num_policies: int,
        network_class: Union[str, Type[PolicyDynamics]],
        network_kwargs: Dict[str, Any] = {},
    ):
        """Constructs `num_networks` instances of the given backend network,
        whose results will be concatenated for the output latent predictions.

        Args:
            num_policies: Number of policies.
            network_class: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__(state_space, action_space)
        self._num_policies = num_policies
        network_class = configs.get_class(network_class, networks)
        self.models = torch.nn.ModuleList(
            [
                network_class(
                    state_space=state_space, action_space=action_space, **network_kwargs
                )
                for _ in range(num_policies)
            ]
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Calls all subnetworks and concatenates the results.

        Computes z' = T_{idx_policy}(z, a).

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
        t_next_latents = torch.cat(next_latents, dim=-1)
        return t_next_latents
