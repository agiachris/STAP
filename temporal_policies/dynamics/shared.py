import pathlib
from typing import Any, Dict, Optional, Sequence, Type, Union

import torch
import numpy as np

from temporal_policies import agents, networks
from temporal_policies.dynamics.latent import LatentDynamics


class SharedDynamics(LatentDynamics):
    """Dynamics model per action with shared latent states.

    We train A dynamics models T_a of the form:

        z^(t+1) = z^(t) + T_a(z^(t), a^(t))

    for every action a..
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
            "network_class": network_class,
            "network_kwargs": network_kwargs,
        }
        assert all(
            policy.state_space == policies[0].state_space for policy in policies[1:]
        )

        super().__init__(
            policies=policies,
            network_class=parent_network_class,
            network_kwargs=parent_network_kwargs,
            state_space=policies[0].state_space,
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
        """Encodes the observation using the first policy's encoder.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.
            policy_args: Auxiliary policy arguments.

        Returns:
            Encoded latent state vector.
        """
        if isinstance(idx_policy, int):
            # [B, O] => [B, Z].
            return self.policies[idx_policy].encoder.encode(observation, policy_args)

        # Assume all encoders are the same.
        return self.policies[0].encoder.encode(observation, policy_args)

        # # [B, O] => [A, B, Z].
        # policy_latents = torch.stack(
        #     [policy.encoder(observation) for policy in self.policies], dim=0
        # )
        #
        # # [B] => [1, B, Z].
        # idx_policy = (
        #     idx_policy.unsqueeze(-1).expand(*policy_latents.shape[1:]).unsqueeze(0)
        # )
        #
        # # [A, B, Z], [1, B, Z] => [B, Z].
        # policy_latent = torch.gather(policy_latents, 0, idx_policy).squeeze(0)
        #
        # return policy_latent
