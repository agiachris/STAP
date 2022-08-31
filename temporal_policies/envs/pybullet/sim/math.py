import math

import dataclasses

from ctrlutils import eigen
import numpy as np


PYBULLET_STEPS_PER_SEC = 240
PYBULLET_TIMESTEP = 1 / PYBULLET_STEPS_PER_SEC


@dataclasses.dataclass
class Pose:
    """6d pose.

    Args:
        pos: 3d position.
        quat: xyzw quaternion.
    """

    pos: np.ndarray = np.zeros(3)
    quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def from_eigen(pose: eigen.Isometry3d) -> "Pose":
        """Creates a pose from an Eigen Isometry3d."""
        pos = np.array(pose.translation)
        quat = np.array(eigen.Quaterniond(pose.linear).coeffs)
        return Pose(pos, quat)

    def to_eigen(self) -> eigen.Isometry3d:
        """Converts a pose to an Eigen Isometry3d."""
        return eigen.Translation3d(self.pos) * eigen.Quaterniond(self.quat)


def comb(n: int, r: int) -> int:
    """Computes (n choose r)."""
    try:
        return math.comb(n, r)
    except AttributeError:
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - 4))
