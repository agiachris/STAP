import pathlib
from typing import Any, Dict, Optional, Sequence, Type, Union

import torch
import numpy as np

from temporal_policies import agents, envs, networks
from temporal_policies.dynamics.latent import LatentDynamics
from temporal_policies.utils import spaces, tensors


class DecoupledDynamics(LatentDynamics):
    """Dynamics model per action per action latent space.

    We train A*A dynamics models T_ab of the form:

        z_a^(t+1) = z_a^(t) + T_ab(z_a^(t), theta_a^(t))

    for every combination of action pairs (a, b).
    """

    def __init__(
        self,
        policies: Sequence[agents.RLAgent],
        network_class: Union[str, Type[networks.dynamics.PolicyDynamics]],
        network_kwargs: Dict[str, Any],
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        device: str = "auto",
    ):
        """Initializes the dynamics model network, dataset, and optimizer.

        Args:
            policies: Ordered list of all policies.
            network_class: Backend network for decoupled dynamics network.
            network_kwargs: Kwargs for network class.
            checkpoint: Dynamics checkpoint.
            device: Torch device.
        """
        parent_network_class = networks.dynamics.Dynamics
        parent_network_kwargs = {
            "policies": policies,
            "network_class": networks.dynamics.ConcatenatedDynamics,
            "network_kwargs": {
                "num_policies": len(policies),
                "network_class": network_class,
                "network_kwargs": network_kwargs,
            },
        }
        state_space = spaces.concatenate_boxes(
            [policy.state_space for policy in policies]
        )

        super().__init__(
            policies=policies,
            network_class=parent_network_class,
            network_kwargs=parent_network_kwargs,
            state_space=state_space,
            action_space=None,
            checkpoint=checkpoint,
            device=device,
        )

    def encode(
        self,
        observation: torch.Tensor,
        idx_policy: Union[int, torch.Tensor],
        policy_args: Union[np.ndarray, Optional[Any]],
    ) -> torch.Tensor:
        """Encodes the observation as a concatenation of latent states for each
        policy.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Concatenated latent state vector of size [Z * A].
        """
        with torch.no_grad():
            zs = [
                policy.encoder.encode(observation, policy_args)
                for policy in self.policies
            ]
            z = torch.cat(zs, dim=-1)
        return z

    @tensors.batch(dims=1)
    def decode(
        self,
        state: torch.Tensor,
        primitive: envs.Primitive,
    ) -> torch.Tensor:
        """Extracts the policy state from the concatenated latent states.

        Args:
            state: Current state.
            primitive: Current primitive.

        Returns:
            Decoded policy state.
        """
        policy_latents = torch.reshape(
            state, (*state.shape[:-1], len(self.policies), -1)
        )
        return policy_latents[:, primitive.idx_policy]
        # if isinstance(idx_policy, int):
        #     return policy_latents[:, idx_policy]
        #
        # idx_policy = (
        #     idx_policy.unsqueeze(-1)
        #     .unsqueeze(-1)
        #     .expand(*idx_policy.shape, 1, policy_latents.shape[-1])
        # )
        # return torch.gather(policy_latents, dim=1, index=idx_policy).squeeze(1)
