from typing import Any, Dict, Optional, Sequence, Type

import torch  # type: ignore


from temporal_policies import agents
from temporal_policies.dynamics import latent as dynamics
from temporal_policies.networks.dynamics import decoupled


class DecoupledDynamics(dynamics.LatentDynamics):
    """Dynamics model per action per action latent space.

    We train A*A dynamics models T_ab of the form:

        z_a^(t+1) = z_a^(t) + T_ab(z_a^(t), theta_a^(t))

    for every combination of action pairs (a, b).
    """

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Type[torch.nn.Module],
        network_kwargs: Dict[str, Any],
        dataset_class: Type[torch.utils.data.IterableDataset],
        dataset_kwargs: Dict[str, Any],
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]],
        scheduler_kwargs: Dict[str, Any],
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
            scheduler_class: Dynamics model learning rate scheduler class.
            scheduler_class: Kwargs for scheduler class.
        """
        network = decoupled.DecoupledDynamicsModel(
            policies=policies,
            network_class=network_class,
            network_kwargs=network_kwargs,
        )
        optimizer = optimizer_class(self.network.parameters(), **optimizer_kwargs)
        if scheduler_class is None:
            scheduler = None
        else:
            scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

        super().__init__(
            policies=policies,
            network=network,
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self._num_policies = len(policies)

    def encode(self, observation: Any, idx_policy: torch.Tensor) -> torch.Tensor:
        """Encodes the observation as a concatenation of latent states for each
        policy.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.

        Returns:
            Concatenated latent state vector of size [Z * A].
        """
        with torch.no_grad():
            zs = [policy.encoder(observation) for policy in self.policies]
            z = torch.cat(zs, dim=-1)
        return z

    def decode(self, latent: torch.Tensor, idx_policy: torch.Tensor) -> Any:
        """Extracts the policy state from the concatenated latent states.

        Args:
            latent: Encoded latent state.
            idx_policy: Index of executed policy.

        Returns:
            Decoded policy state.
        """
        policy_latents = torch.reshape(
            latent, (*latent.shape[:-1], self._num_policies, -1)
        )
        idx_policy = (
            idx_policy.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*idx_policy.shape, 1, policy_latents.shape[-1])
        )
        return torch.gather(policy_latents, dim=1, index=idx_policy).squeeze(1)
