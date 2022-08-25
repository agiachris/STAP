import dataclasses
from typing import Any, Dict

from ctrlutils import eigen
import numpy as np


PYBULLET_TIMESTEP = 1 / 240


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

    def from_dict(self, pose: Dict[str, Any]) -> "Pose":
        """Creates a pose from dict format."""
        return Pose(
            np.array(pose["pose"]),
            np.array(
                [pose["ori"]["x"], pose["ori"]["y"], pose["ori"]["z"], pose["ori"]["w"]]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts a pose to dict format."""
        return {
            "pos": self.pos.tolist(),
            "ori": {
                "x": self.quat[0],
                "y": self.quat[1],
                "z": self.quat[2],
                "w": self.quat[3],
            },
        }
