from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import articulated_body, math


class Gripper(articulated_body.ArticulatedBody):
    def __init__(
        self,
        physics_id: int,
        body_id: int,
        torque_joints: List[str],
        position_joints: List[str],
        finger_links: List[str],
        base_link: str,
        command_multipliers: List[float],
        inertia_kwargs: Dict[str, Any],
        pos_gains: Tuple[float, float],
        pos_threshold: float,
        timeout: float,
    ):
        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            torque_joints=torque_joints,
            position_joints=position_joints,
            timeout=timeout,
        )

        cmd_multipliers = np.array(command_multipliers, dtype=np.float64)
        self._torque_multipliers = cmd_multipliers[: len(self.torque_joints)]
        self._position_multipliers = cmd_multipliers[len(self.torque_joints) :]

        def get_link_mass(joint_id: int):
            return p.getDynamicsInfo(
                self.body_id, joint_id, physicsClientId=self.physics_id
            )[0]

        self._masses = np.array(
            [get_link_mass(joint_id) for joint_id in self.torque_joints]
        )
        self._inertia = dyn.SpatialInertiad(
            inertia_kwargs["mass"],
            np.array(inertia_kwargs["com"]),
            np.array(inertia_kwargs["inertia"]),
        )

        link_ids = {self.link(joint_id).name: joint_id for joint_id in range(self.dof)}
        self._finger_links = [link_ids[link_name] for link_name in finger_links]
        self._base_link = link_ids[base_link]

        self.pos_gains = pos_gains
        self.pos_threshold = pos_threshold

        self._torque_control = False
        self._grasp_constraint_id: Optional[int] = None

        self._q_home, _ = self.get_state(self.joints)

    @property
    def inertia(self) -> dyn.SpatialInertiad:
        return self._inertia

    @property
    def finger_links(self) -> List[int]:
        return self._finger_links

    @property
    def base_link(self) -> int:
        return self._base_link

    def reset(self) -> bool:
        self.remove_grasp_constraint()
        self.reset_joints(self._q_home, self.joints)
        self.freeze_grasp()
        return True

    def is_object_grasped(self, body_id: int) -> bool:
        for finger_link_id in self.finger_links:
            contacts = p.getContactPoints(
                bodyA=self.body_id,
                bodyB=body_id,
                linkIndexA=finger_link_id,
                physicsClientId=self.physics_id,
            )
            if not contacts:
                return False

            # Contact positions/normals on the object in world frame.
            contact_normals = np.array([c[7] for c in contacts])

            if (np.abs(contact_normals[:, 2]) > 1 / np.sqrt(3)).any():
                return False

        return True

    def create_grasp_constraint(self, body_id: int) -> bool:
        self.remove_grasp_constraint()
        if not self.is_object_grasped(body_id):
            return False

        T_body_to_world = math.Pose(
            *p.getBasePositionAndOrientation(body_id, physicsClientId=self.physics_id)
        ).to_eigen()
        T_ee_to_world = self.link(self._base_link).pose().to_eigen()
        T_body_to_ee = T_ee_to_world.inverse() * T_body_to_world

        self._grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=self._base_link,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=np.zeros(3),
            parentFramePosition=T_body_to_ee.translation,
            childFramePosition=np.zeros(3),
            parentFrameOrientation=eigen.Quaterniond(T_body_to_ee.linear).coeffs,
            physicsClientId=self.physics_id,
        )
        return True

    def remove_grasp_constraint(self):
        if self._grasp_constraint_id is None:
            return

        p.removeConstraint(self._grasp_constraint_id, physicsClientId=self.physics_id)
        self._grasp_constraint_id = None

    def set_grasp(
        self,
        command: float,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._command = command
        if timeout is None:
            timeout = self.timeout
        self._pos_gains = self.pos_gains if pos_gains is None else pos_gains

        self._dq_avg = 1.0
        self._iter_timeout = int(timeout / math.PYBULLET_TIMESTEP)

        self._torque_control = True

    def freeze_grasp(self) -> None:
        self._torque_control = False
        q = self.get_state(self.joints)[0]
        self.apply_positions(q, self.joints)

    def update_torques(self) -> articulated_body.ControlStatus:
        if not self._torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        joint_states = p.getJointStates(
            self.body_id, self._torque_joints, physicsClientId=self.physics_id
        )
        q, dq, _, _ = zip(*joint_states)
        q = np.array(q)
        dq = np.array(dq)
        q_des = self._torque_multipliers * self._command
        q_err = q - q_des

        # Compute commands.
        ddq = -self._pos_gains[0] * q_err - self._pos_gains[1] * dq
        tau = self._masses * ddq
        current = q[0] / self._torque_multipliers[0]
        q_command = self._position_multipliers * current

        self.apply_torques(tau)
        self.apply_positions(q_command)

        self._dq_avg = 0.5 * abs(dq[0]) + 0.5 * self._dq_avg
        self._iter_timeout -= 1

        # Return position converged.
        if q_err.dot(q_err) < self.pos_threshold * self.pos_threshold:
            return articulated_body.ControlStatus.POS_CONVERGED

        # Return velocity converged.
        if self._dq_avg < 0.0001:
            return articulated_body.ControlStatus.VEL_CONVERGED

        # Return timeout.
        if self._iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT

        return articulated_body.ControlStatus.IN_PROGRESS
