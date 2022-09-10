from typing import Optional, Union

from ctrlutils import eigen
import numpy as np

from temporal_policies.envs.pybullet.sim import math


class ObjectState:
    RANGES = {
        "x": (-0.3, 0.9),
        "y": (-0.5, 0.5),
        "z": (-0.1, 0.5),
        "wx": (-np.pi, np.pi),
        "wy": (-np.pi, np.pi),
        "wz": (-np.pi, np.pi),
        "box_size_x": (0.0, 0.1),
        "box_size_y": (0.0, 0.1),
        "box_size_z": (0.0, 0.2),
        "head_length": (0.0, 0.3),
        "handle_length": (0.0, 0.5),
        "handle_y": (-1.0, 1.0),
    }

    def __init__(self, vector: Optional[np.ndarray] = None):
        if vector is None:
            vector = np.zeros(len(self.RANGES), dtype=np.float32)
        elif vector.shape[-1] != len(self.RANGES):
            vector = vector.reshape(
                (
                    *vector.shape[:-1],
                    vector.shape[-1] // len(self.RANGES),
                    len(self.RANGES),
                )
            )
        self.vector = vector

    @property
    def pos(self) -> np.ndarray:
        return self.vector[..., :3]

    @pos.setter
    def pos(self, pos: np.ndarray) -> None:
        self.vector[..., :3] = pos

    @property
    def aa(self) -> np.ndarray:
        return self.vector[..., 3:6]

    @aa.setter
    def aa(self, aa: np.ndarray) -> None:
        self.vector[..., 3:6] = aa

    @property
    def box_size(self) -> np.ndarray:
        return self.vector[..., 6:9]

    @box_size.setter
    def box_size(self, box_size: np.ndarray) -> None:
        self.vector[..., 6:9] = box_size

    @property
    def head_length(self) -> Union[float, np.ndarray]:
        if self.vector.ndim > 1:
            return self.vector[..., 9:10]
        return self.vector[9]

    @head_length.setter
    def head_length(self, head_length: Union[float, np.ndarray]) -> None:
        self.vector[..., 9:10] = head_length

    @property
    def handle_length(self) -> Union[float, np.ndarray]:
        if self.vector.ndim > 1:
            return self.vector[..., 10:11]
        return self.vector[10]

    @handle_length.setter
    def handle_length(self, handle_length: Union[float, np.ndarray]) -> None:
        self.vector[..., 10:11] = handle_length

    @property
    def handle_y(self) -> Union[float, np.ndarray]:
        if self.vector.ndim > 1:
            return self.vector[..., 11:12]
        return self.vector[11]

    @handle_y.setter
    def handle_y(self, handle_y: Union[float, np.ndarray]) -> None:
        self.vector[..., 11:12] = handle_y

    @classmethod
    def range(cls) -> np.ndarray:
        return np.array(list(cls.RANGES.values()), dtype=np.float32).T

    def pose(self) -> math.Pose:
        angle = np.linalg.norm(self.aa)
        if angle == 0:
            quat = eigen.Quaterniond.identity()
        else:
            axis = self.aa / angle
            quat = eigen.Quaterniond(eigen.AngleAxisd(angle, axis))
        return math.Pose(pos=self.pos, quat=quat.coeffs)

    def set_pose(self, pose: math.Pose) -> None:
        aa = eigen.AngleAxisd(eigen.Quaterniond(pose.quat))
        self.pos = pose.pos
        self.aa = aa.angle * aa.axis

    def __repr__(self) -> str:
        return (
            "{\n"
            f"    pos: {self.pos},\n"
            f"    aa: {self.aa},\n"
            f"    box_size: {self.box_size},\n"
            f"    head_length: {self.head_length},\n"
            f"    handle_length: {self.handle_length},\n"
            f"    handle_y: {self.handle_y},\n"
            "}"
        )
