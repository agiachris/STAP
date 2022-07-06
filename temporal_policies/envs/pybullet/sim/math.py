import dataclasses
from typing import Any, Dict

from ctrlutils import eigen
import numpy as np


PYBULLET_TIMESTEP = 1 / 240


@dataclasses.dataclass
class Pose:
    pos: np.ndarray = np.zeros(3)
    quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def from_eigen(pose: eigen.Isometry3d) -> "Pose":
        pos = np.array(pose.translation)
        quat = np.array(eigen.Quaterniond(pose.linear).coeffs)
        return Pose(pos, quat)

    def to_eigen(self) -> eigen.Isometry3d:
        return eigen.Translation3d(self.pos) * eigen.Quaterniond(self.quat)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pos": self.pos.tolist(),
            "ori": {
                "x": self.quat[0],
                "y": self.quat[1],
                "z": self.quat[2],
                "w": self.quat[3],
            }
        }
