from typing import Optional, Tuple

import torch

from temporal_policies import envs, networks, scod
from temporal_policies.agents import base, wrapper


class SCODCritic(networks.critics.Critic):
    def __init__(self, scod_wrapper: scod.WrapperSCOD):
        self.scod_wrapper = scod_wrapper

    def forward(  # type: ignore
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.scod_wrapper.forward(state, action)

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.scod_wrapper.predict(state, action)


class SCODCriticAgent(wrapper.WrapperAgent):
    """Agent wrapper that returns an uncertainty-transformed metric over the critic."""

    def __init__(
        self,
        policy: base.Agent,
        scod_wrapper: scod.WrapperSCOD,
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Constructs the random agent.

        Args:
            policy: Main policy whose predictions are used as the mean.
            scod_wrapper: SCOD wrapper around the critic.
            env: Policy env (unused, but included for API consistency).
            std: Standard deviation.
            device: Torch device.
        """
        super().__init__(
            state_space=policy.state_space,
            action_space=policy.action_space,
            observation_space=policy.observation_space,
            actor=policy.actor,
            critic=SCODCritic(scod_wrapper),
            encoder=policy.encoder,
            device=device,
        )
