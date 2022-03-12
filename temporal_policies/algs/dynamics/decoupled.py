from typing import Any, Dict, List, Type

import torch  # type: ignore


from temporal_policies.algs import base as algorithms
from temporal_policies.algs.dynamics import base as dynamics


class ConcatenatedDynamicsModel(torch.nn.Module):
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
            network: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__()
        self._num_policies = num_policies
        self.models = torch.nn.ModuleList(
            [network_class(**network_kwargs) for _ in range(num_policies)]
        )

    def forward(
        self, latent: torch.Tensor, policy_params: torch.Tensor
    ) -> torch.Tensor:
        """Calls all subnetworks and concatenates the results.

        Computes z' = T_a(z, theta_a).

        Args:
            latent: Current latent state.
            policy_params: Policy parameters (for a single policy).

        Returns:
            Concatenated latent prediction z' for a single policy action.
        """
        policy_latents = torch.reshape(latent, (self._num_policies, -1))
        next_latents = [
            model_a(policy_latent, policy_params)
            for policy_latent, model_a in zip(policy_latents, self.models)
        ]
        next_latents = torch.cat(next_latents, dim=-1)
        return next_latents


# TODO: Reuse for unified dynamics model.
class DecoupledDynamicsModel(torch.nn.Module):
    """Network module for the decoupled dynamics model."""

    def __init__(
        self,
        policies: List[algorithms.Algorithm],
        network_class: Type[torch.nn.Module],
        network_kwargs: Dict[str, Any],
    ):
        """Constructs `num_networks` instances of ConcatenatedDynamicsModel.

        Args:
            num_policies: Number of policies.
            network: Backend network.
            network_kwargs: Backend network arguments.
        """
        super().__init__()

        self._num_policies = len(policies)
        self.models = torch.nn.ModuleList(
            [
                ConcatenatedDynamicsModel(
                    self._num_policies,
                    network_class,
                    dict(
                        network_kwargs,
                        action_space=policy.env.action_space,
                    ),
                )
                for policy in policies
            ]
        )

    def forward(
        self, latents: torch.Tensor, policy_indices: torch.Tensor, policy_params: Any
    ) -> torch.Tensor:
        """Predicts the next latent state using separate dynamics model per
        action.

        Args:
            latents: Current latent state.
            policy_indices: Index of executed policy.
            policy_params: Policy parameters.

        Returns:
            Prediction of next latent state.
        """
        if latents.dim == 1:
            assert policy_indices.numel == 1

            next_latent = self.models[policy_indices](latents, policy_params)
        else:
            assert latents.shape[0] == policy_indices.shape[0]
            assert policy_indices.shape[0] == len(policy_params)

            next_latents = [
                self.models[idx_policy](latent, params)
                for latent, idx_policy, params in zip(
                    latents, policy_indices, policy_params
                )
            ]
            next_latent = torch.stack(next_latents, dim=0)

        return next_latent


class DecoupledDynamics(dynamics.DynamicsModel):
    """Dynamics model per action per action latent space.

    We train A*A dynamics models T_ab of the form:

        z_a^(t+1) = z_a^(t) + T_ab(z_a^(t), theta_a^(t))

    for every combination of action pairs (a, b).
    """

    def __init__(
        self,
        policies: List[algorithms.Algorithm],
        network_class: Type[torch.nn.Module],
        network_kwargs: Dict[str, Any],
        dataset_class: Type[torch.utils.data.IterableDataset],
        dataset_kwargs: Dict[str, Any],
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Backend network for decoupled dynamics network.
            network_kwargs: Kwargs for network class.
            dataset_class: Dynamics model dataset class.
            dataset_kwargs: Kwargs for dataset class.
            optimizer_class: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
        """
        super().__init__(
            policies=policies,
            network_class=DecoupledDynamicsModel,
            network_kwargs={
                "policies": policies,
                "network_class": network_class,
                "network_kwargs": network_kwargs,
            },
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def encode(self, observation: Any, idx_policy: torch.Tensor) -> torch.Tensor:
        """Encodes the observation as a concatenation of latent states for each
        policy.

        Args:
            observation: Common observation across all policies.
            idx_policy: Unused.

        Returns:
            Concatenated latent state vector of size [Z * A].
        """
        zs = [policy.network.encoder(observation) for policy in self.policies]
        z = torch.cat(zs, dim=-1)
        return z
