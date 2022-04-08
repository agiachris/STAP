import pathlib
from typing import Any, Dict, Optional, Sequence, Type, Union

import torch  # type: ignore


from temporal_policies import agents, networks
from temporal_policies.dynamics import latent as dynamics
from temporal_policies.utils import spaces


class DecoupledDynamics(dynamics.LatentDynamics):
    """Dynamics model per action per action latent space.

    We train A*A dynamics models T_ab of the form:

        z_a^(t+1) = z_a^(t) + T_ab(z_a^(t), theta_a^(t))

    for every combination of action pairs (a, b).
    """

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[torch.nn.Module]],
        network_kwargs: Dict[str, Any],
        dataset_class: Union[str, Type[torch.utils.data.IterableDataset]],
        dataset_kwargs: Dict[str, Any],
        optimizer_class: Union[str, Type[torch.optim.Optimizer]],
        optimizer_kwargs: Dict[str, Any],
        scheduler_class: Optional[
            Union[str, Type[torch.optim.lr_scheduler._LRScheduler]]
        ] = None,
        scheduler_kwargs: Dict[str, Any] = {},
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Backend network for decoupled dynamics network.
            network_kwargs: Kwargs for network class.
            dataset: Dynamics model dataset class or class name.
            dataset_kwargs: Kwargs for dataset class.
            optimizer: Dynamics model optimizer class.
            optimizer_kwargs: Kwargs for optimizer class.
            scheduler: Optional dynamics model learning rate scheduler class.
            scheduler_kwargs: Kwargs for scheduler class.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        parent_network_class = networks.dynamics.DecoupledDynamics
        parent_network_kwargs = {
            "policies": policies,
            "network_class": network_class,
            "network_kwargs": network_kwargs,
        }
        state_space = spaces.concatenate_boxes(
            [policy.state_space for policy in policies]
        )

        super().__init__(
            policies=policies,
            network_class=parent_network_class,
            network_kwargs=parent_network_kwargs,
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            state_space=state_space,
            action_space=None,
            checkpoint=checkpoint,
            device=device,
        )

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

    def decode(
        self,
        latent: torch.Tensor,
        idx_policy: torch.Tensor,
        policy_args: Optional[Any] = None,
    ) -> Any:
        """Extracts the policy state from the concatenated latent states.

        Args:
            latent: Encoded latent state.
            idx_policy: Index of executed policy.

        Returns:
            Decoded policy state.
        """
        policy_latents = torch.reshape(
            latent, (*latent.shape[:-1], len(self.policies), -1)
        )
        if isinstance(idx_policy, int):
            return policy_latents[:, idx_policy]

        idx_policy = (
            idx_policy.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*idx_policy.shape, 1, policy_latents.shape[-1])
        )
        return torch.gather(policy_latents, dim=1, index=idx_policy).squeeze(1)
