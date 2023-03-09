from typing import List, Any, Optional

import abc
import torch

from temporal_policies.networks.critics.base import Critic
from temporal_policies.networks.critics.mlp import ContinuousMLPCritic


class ContinuousEnsembleCritic(Critic, abc.ABC):
    def __init__(
        self,
        critic: ContinuousMLPCritic,
        pessimistic: bool,
        clip: bool,
    ):
        """Construct ContinuousEnsembleCritic.

        Args:
            critic: Base Critic.
            pessimistic: Estimated rewards from min(Qi) instead of mean(Qi).
            clip: Clip Q-values between [0, 1].
        """
        assert isinstance(critic, ContinuousMLPCritic) and len(critic.qs) > 1
        super().__init__()
        self.network = critic
        self._pessimistic = pessimistic
        self._clip = clip

    @property
    def pessimistic(self) -> bool:
        return self._pessimistic

    @property
    def clip(self) -> bool:
        return self._clip

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        return self.network.forward(state, action)

    @abc.abstractmethod
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Predict must be implemented in subclasses.")


class EnsembleLCBCritic(ContinuousEnsembleCritic):
    def __init__(self, scale: float, **kwargs: Any):
        """Construct EnsembleLCBCritic.

        Args:
            scale: Lower confidence bound (LCB) scale factor, <min/mean>(Qi) - scale * std(Qi).
        """
        assert isinstance(scale, float)
        super().__init__(**kwargs)
        self.scale = scale

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the LCB Q-value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            LCB of Q-value.
        """
        qs: torch.Tensor = torch.stack(self.forward(state, action))
        q = torch.min(qs, dim=0).values if self.pessimistic else qs.mean(dim=0)
        q -= self.scale * qs.std(dim=0)
        return torch.clamp(q, 0, 1) if self.clip else q


class EnsembleThresholdCritic(ContinuousEnsembleCritic):
    def __init__(self, threshold: float, value: float, **kwargs: Any):
        """Construct EnsembleThresholdCritic.

        Args:
            threshold: Out-of-distribution threshold on std(Qi).
            value: Value assignment to out-of-distribution detected sample.
        """
        assert isinstance(threshold, float) and threshold >= 0.0
        assert isinstance(value, float) and value >= 0.0
        super().__init__(**kwargs)
        self.threshold = threshold
        self.value = value

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            OOD thresholded Q-values.
        """
        qs: torch.Tensor = torch.stack(self.forward(state, action))
        q = torch.min(qs, dim=0).values if self.pessimistic else qs.mean(dim=0)
        q[qs.std(dim=0) > self.threshold] = self.value
        return torch.clamp(q, 0, 1) if self.clip else q


class EnsembleOODCritic(ContinuousEnsembleCritic):
    def __init__(self, threshold: float, **kwargs: Any):
        """Construct EnsembleOODCritic.

        Args:
            threshold: Out-of-distribution threshold on std(Qi).
        """
        assert isinstance(threshold, float) and threshold >= 0.0
        super().__init__(**kwargs)
        self.threshold = threshold
        self._cached_detections: Optional[torch.Tensor] = None

    @property
    def detect(self) -> torch.Tensor:
        """Returns tensor of OOD detections."""
        if self._cached_detections is None:
            raise ValueError("Must call EnsembleOODCritic.predict before detect.")
        detections = self._cached_detections
        self._cached_detections = None
        return detections

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Q-values.
        """
        qs: torch.Tensor = torch.stack(self.forward(state, action))
        self._cached_detections = (qs.std(dim=0) > self.threshold).bool().detach().cpu()
        q = torch.min(qs, dim=0).values if self.pessimistic else qs.mean(dim=0)
        return torch.clamp(q, 0, 1) if self.clip else q
