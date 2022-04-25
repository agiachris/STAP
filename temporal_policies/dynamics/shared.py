import pathlib
from typing import Any, Dict, Optional, Sequence, Type, Union

import torch  # type: ignore


from temporal_policies import agents, networks
from temporal_policies.dynamics.latent import LatentDynamics
from temporal_policies.utils.typing import ObsType


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
        self, observation: ObsType, idx_policy: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Encodes the observation using the first policy's encoder.

        Args:
            observation: Common observation across all policies.
            idx_policy: Index of executed policy.

        Returns:
            Encoded latent state vector.
        """
        # Assume all encoders are the same.
        # [B, O] => [B, Z].
        policy_latent = self.policies[0].encoder(observation)

        return policy_latent
