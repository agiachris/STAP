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
        """Body aabb.

        Note: The aabb given by Pybullet is larger than the true aabb for
        collision detection purposes.

        Also, the aabb is only reported for the object base.
        """
        return np.array(p.getAABB(self.body_id, physicsClientId=self.physics_id))

    def pose(self) -> math.Pose:
        """Base pose."""
        pos, quat = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.physics_id
        )
        return math.Pose(np.array(pos), np.array(quat))

    def set_pose(self, pose: math.Pose) -> None:
        """Sets the base pose."""
        p.resetBasePositionAndOrientation(
            self.body_id, pose.pos, pose.quat, physicsClientId=self.physics_id
        )

    def twist(self) -> np.ndarray:
        """Base twist."""
        v, w = p.getBaseVelocity(self.body_id, physicsClientId=self.physics_id)
        return np.concatenate([v, w])

    @property
    def dof(self) -> int:
        """Total number of joints in the articulated body, including non-controlled joints."""
        try:
            return self._dof  # type: ignore
        except AttributeError:
            pass

        self._dof = p.getNumJoints(self.body_id, physicsClientId=self.physics_id)
        return self._dof

    @property
    def inertia(self) -> dyn.SpatialInertiad:
        """Base inertia."""
        try:
            return self._inertia  # type: ignore
        except AttributeError:
            pass

        was_frozen = self.unfreeze()
        dynamics_info = p.getDynamicsInfo(
            self.body_id, -1, physicsClientId=self.physics_id
        )
        if was_frozen:
            self.freeze()

        mass = dynamics_info[0]
        inertia_xyz = dynamics_info[2]
        com = np.array(dynamics_info[3])
        quat_inertia = eigen.Quaterniond(dynamics_info[4])
        T_inertia = eigen.Translation3d.identity() * quat_inertia
        self._inertia = (
            dyn.SpatialInertiad(mass, com, np.concatenate([inertia_xyz, np.zeros(3)]))
            * T_inertia
        )

        return self._inertia

    def freeze(self) -> bool:
        """Disable simulation for this body.

        Returns:
            Whether the object's frozen status changed.
        """
        if not hasattr(self, "_is_frozen"):
            self._mass = p.getDynamicsInfo(
                self.body_id, -1, physicsClientId=self.physics_id
            )[0]
        elif self._is_frozen:  # type: ignore
            return False

        p.changeDynamics(self.body_id, -1, mass=0, physicsClientId=self.physics_id)
        self._is_frozen = True
        return True

    def unfreeze(self) -> bool:
        """Enable simulation for this body.

        Returns:
            Whether the object's frozen status changed.
        """
        if not hasattr(self, "_is_frozen") or not self._is_frozen:
            return False

        p.changeDynamics(
            self.body_id, -1, mass=self._mass, physicsClientId=self.physics_id
        )
        self._is_frozen = False
        return True


@dataclasses.dataclass
class Link:
    physics_id: int
    body_id: int
    link_id: int

    @property
    def name(self) -> str:
        """Link name."""
        try:
            return self._name  # type: ignore
        except AttributeError:
            pass

        self._name = p.getJointInfo(
            self.body_id, self.link_id, physicsClientId=self.physics_id
        )[12].decode("utf8")
        return self._name

    def pose(self) -> math.Pose:
        """World pose of the center of mass."""
        pos, quat = p.getLinkState(
            self.body_id, self.link_id, physicsClientId=self.physics_id
        )[:2]
        return math.Pose(np.array(pos), np.array(quat))

    @property
    def inertia(self) -> str:
        """Inertia at the center of mass frame."""
        try:
            return self._inertia  # type: ignore
        except AttributeError:
            pass

        dynamics_info = p.getDynamicsInfo(
            self.body_id, self.link_id, physicsClientId=self.physics_id
        )
        mass = dynamics_info[0]
        inertia_xyz = dynamics_info[2]
        com = np.zeros(3)
        self._inertia = dyn.SpatialInertiad(
            mass, com, np.concatenate([inertia_xyz, np.zeros(3)])
        )

        return self._inertia

    @property
    def joint_limits(self) -> np.ndarray:
        """(lower, upper) joint limits."""
        try:
            return self._joint_limits  # type: ignore
        except AttributeError:
            pass

        joint_info = p.getJointInfo(
            self.body_id, self.link_id, physicsClientId=self.physics_id
        )
        self._joint_limits = np.array(joint_info[8:10])

        return self._joint_limits
