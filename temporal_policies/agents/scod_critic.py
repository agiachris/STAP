from typing import Optional, Tuple

import torch

from temporal_policies import envs, networks
from temporal_policies.agents import base, wrapper, scod


class SCODCritic(networks.critics.Critic):
    def __init__(self, scod_wrapper: scod.WrapperSCOD):
        super().__init__()
        self.scod_wrapper = scod_wrapper

    def forward(  # type: ignore
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.scod_wrapper.forward(state, action)

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.scod_wrapper.predict(state, action)


class SCODProbabilisticCritic(networks.critics.ProbabilisticCritic):
    def __init__(self, scod_wrapper: scod.WrapperSCOD):
        super().__init__()
        self.scod_wrapper = scod_wrapper

    def forward(  # type: ignore
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.distributions.Distribution:
        loc, var, _ = self.scod_wrapper.forward(state, action)
        if loc.size(-1) == 1:
            loc = loc.squeeze(-1)
        if var.size(-1) == 1:
            var = var.squeeze(-1)
        return torch.distributions.Normal(loc=loc, scale=torch.sqrt(var))

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
        """Constructs the SCODCriticAgent.

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


class SCODProbabilisticCriticAgent(wrapper.WrapperAgent):
    """Agent wrapper that returns a scod-informed distribution over the critic."""

    def __init__(
        self,
        policy: base.Agent,
        scod_wrapper: scod.WrapperSCOD,
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Constructs the SCODProbabilisticCriticAgent.

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
            critic=SCODProbabilisticCritic(scod_wrapper),
            encoder=policy.encoder,
            device=device,
        )
