import dataclasses

from ctrlutils import eigen
import spatialdyn as dyn
import numpy as np
import pybullet as p

from temporal_policies.envs.pybullet.sim import math


@dataclasses.dataclass
class Body:
    physics_id: int
    body_id: int

    def aabb(self) -> np.ndarray:
        return np.array(p.getAABB(self.body_id, physicsClientId=self.physics_id))

    def pose(self) -> math.Pose:
        pos, quat = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.physics_id
        )
        return math.Pose(pos, quat)

    def set_pose(self, pose: math.Pose) -> None:
        p.resetBasePositionAndOrientation(
            self.body_id, pose.pos, pose.quat, physicsClientId=self.physics_id
        )

    def twist(self) -> np.ndarray:
        v, w = p.getBaseVelocity(self.body_id, physicsClientId=self.physics_id)
        return np.concatenate([v, w])

    @property
    def inertia(self) -> dyn.SpatialInertiad:
        try:
            return self._inertia  # type: ignore
        except AttributeError:
            dynamics_info = p.getDynamicsInfo(
                self.body_id, -1, physicsClientId=self.physics_id
            )
            mass = dynamics_info[0]
            inertia_xyz = dynamics_info[2]
            com = np.array(dynamics_info[3])
            quat_inertia = eigen.Quaterniond(dynamics_info[4])
            T_inertia = eigen.Translation3d.identity() * quat_inertia
            inertia = dyn.SpatialInertiad(
                mass, com, np.concatenate([inertia_xyz, np.zeros(3)])
            )
            self._inertia = inertia * T_inertia
            return self._inertia


@dataclasses.dataclass
class Link:
    physics_id: int
    body_id: int
    link_id: int

    @property
    def name(self) -> str:
        try:
            return self._name  # type: ignore
        except AttributeError:
            self._name = p.getJointInfo(
                self.body_id, self.link_id, physicsClientId=self.physics_id
            )[12].decode("utf8")
            return self._name

    def pose(self) -> math.Pose:
        pos, quat = p.getLinkState(
            self.body_id, self.link_id, physicsClientId=self.physics_id
        )[:2]
        return math.Pose(pos, quat)
