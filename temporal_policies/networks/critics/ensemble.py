from typing import List, Any, Optional, Callable, Dict

import abc
import torch
from torch.utils.hooks import RemovableHandle

from temporal_policies.networks.critics.base import Critic
from temporal_policies.networks.critics.mlp import ContinuousMLPCritic, MLP


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
        self._network = critic
        self._pessimistic = pessimistic
        self._clip = clip

    @property
    def network(self) -> ContinuousMLPCritic:
        return self._network

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
        self._scale = scale
    
    @property
    def scale(self) -> float:
        return self._scale

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
        self._threshold = threshold
        self._value = value

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def value(self) -> float:
        return self._value

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


class EnsembleDetectorCritic(ContinuousEnsembleCritic, abc.ABC):
    def __init__(self, threshold: float, **kwargs: Any):
        """Construct EnsembleDetectorCritic.

        Args:
            threshold: Out-of-distribution threshold on std(Qi).
        """
        assert isinstance(threshold, float) and threshold >= 0.0
        super().__init__(**kwargs)
        self._threshold = threshold
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @abc.abstractproperty
    def detect(self) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclasses.")


class EnsembleOODCritic(EnsembleDetectorCritic):
    def __init__(self, threshold: float, **kwargs: Any):
        """Construct EnsembleOODCritic.

        Args:
            threshold: Out-of-distribution threshold on std(Qi).
        """
        super().__init__(threshold, **kwargs)
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
        q = torch.min(qs, dim=0).values if self.pessimistic else qs.mean(dim=0)
        self._cached_detections = (qs.std(dim=0) > self.threshold).bool().detach().cpu()
        return torch.clamp(q, 0, 1) if self.clip else q


class EnsembleLogitOODCritic(EnsembleDetectorCritic):
    def __init__(self, threshold: float, **kwargs: Any):
        """Construct EnsembleLogitOODCritic.

        Args:
            threshold: Out-of-distribution threshold on std(Qi).
        """
        super().__init__(threshold, **kwargs)
        self._activations: Dict[str, torch.Tensor] = {}
        self._hook_handles: Dict[str, RemovableHandle] = {}
        self._reset_hooks()
        
    def _create_forward_hook(self, key: str) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
        """Return forward hook callable."""
        def hook(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self._activations[key] = output.squeeze(-1).detach()
        return hook

    def _reset_hooks(self) -> None:
        """Set forward hooks before last layer of each Q-network."""
        for hook_handle in self._hook_handles.values():
            hook_handle.remove()

        self._activations: Dict[str, torch.Tensor] = {}
        self._hook_handles: Dict[str, RemovableHandle] = {}

        for idx, q in enumerate(self.network.qs):
            if (isinstance(q, MLP) 
                and isinstance(q.net[-1], torch.nn.Sigmoid) 
                and isinstance(q.net[-2], torch.nn.Linear)
            ):
                hook = self._create_forward_hook(f"q{idx}")
                hook_handle = q.net[-2].register_forward_hook(hook)
                self._hook_handles[f"q{idx}"] = hook_handle
            else:
                raise ValueError(f"Require Q-networks with Sigmoid output activation.")

    @property
    def logits(self) -> torch.Tensor:
        """Returns logits stored during forward pass."""
        if not self._activations:
            raise ValueError("Must call EnsembleLogitOODCritic.predict before logits.")
        return torch.stack(list(self._activations.values()))

    @property
    def detect(self) -> torch.Tensor:
        """Returns tensor of OOD detections."""
        detections = (self.logits.std(dim=0) > self.threshold).bool().cpu()
        self._reset_hooks()
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
        q = torch.min(qs, dim=0).values if self.pessimistic else qs.mean(dim=0)
        return torch.clamp(q, 0, 1) if self.clip else q
