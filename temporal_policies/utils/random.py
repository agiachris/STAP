import random
from typing import Optional

import numpy as np  # type: ignore
import torch  # type: ignore


def seed(n: Optional[int]) -> None:
    """Sets the random seed.

    Args:
        n: Optional seed. If None, no seed is set.
    """
    if n is None:
        return

    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)
