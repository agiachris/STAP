from typing import Dict, List

import torch


class DummyScheduler:
    """Dummy scheduler class for trainers."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._lr = optimizer.defaults["lr"]

    def step(self) -> None:
        pass

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    def get_last_lr(self) -> List[float]:
        return [self._lr]
