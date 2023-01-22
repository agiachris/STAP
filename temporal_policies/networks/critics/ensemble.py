from typing import List

import torch

from temporal_policies.networks.critics.base import Critic
from temporal_policies.networks.critics.mlp import ContinuousMLPCritic


class ContinuousEnsembleCritic(Critic):
    def __init__(
        self,
        critic: ContinuousMLPCritic,
        scale: float,
        clip: bool,
    ):
        """Construct ContinuousEnsembleCritic
        
        Args:
            critic: Base Critic.
            scale: Lower-confidence bound scale.
        """
        assert isinstance(critic, ContinuousMLPCritic) and len(critic.qs) > 1
        super().__init__()
        self.network = critic
        self.scale = scale
        self.clip = clip

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        return self.network.forward(state, action)

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Lower-confidence bound of Q-value.
        """
        qs: torch.Tensor = torch.stack(self.forward(state, action))
        q = qs.mean(dim=0) - self.scale * qs.std(dim=0)
        return torch.clamp(q, 0, 1) if self.clip else q
