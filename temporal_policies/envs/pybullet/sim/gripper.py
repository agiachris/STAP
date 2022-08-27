import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import articulated_body, body, math


@dataclasses.dataclass
class GripperState:
    """Mutable gripper state."""

    command: float = 0.0
    torque_control: bool = False
    grasp_constraint_id: Optional[int] = None
    grasp_body_id: Optional[int] = None
    grasp_T_body_to_ee: Optional[math.Pose] = None
    dq_avg = 0.0
    iter_timeout = 0


class Gripper(articulated_body.ArticulatedBody):
    """Gripper controlled with torque control."""

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        T_world_to_ee: eigen.Isometry3d,
        torque_joints: List[str],
        position_joints: List[str],
        finger_links: List[str],
        base_link: str,
        command_multipliers: List[float],
        finger_contact_normals: List[List[float]],
        inertia_kwargs: Dict[str, Any],
        pos_gains: Tuple[float, float],
        pos_threshold: float,
        timeout: float,
    ):
        """Constructs the arm from yaml config.

        Args:
            physics_id: Pybullet physics client id.
            body_id: Pybullet body id.
            torque_joints: List of torque_controlled joint names.
            position_joints: List of position-controlled joint names.
            finger_links: Finger link names.
            base_link: Gripper base link name.
            command_multipliers: Conversion from [0.0, 1.0] grasp command to
                corresponding joint position.
            finger_contact_normals: Direction of the expected contact normal for
                each of the finger links in the finger link frame, pointing
                towards the grasped object.
            inertia_kwargs: Gripper inertia kwargs ({"mass": Float, "com":
                List[Float, 3], "inertia": List[Float, 6]}).
            pos_gains: (kp, kv) position gains.
            pos_threshold: (position, velocity) error threshold for position convergence.
            timeout: Default command timeout.
        """
        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            torque_joints=torque_joints,
            position_joints=position_joints,
            timeout=timeout,
        )

        self._command_multipliers = np.array(command_multipliers, dtype=np.float64)
        self._torque_multipliers = self._command_multipliers[: len(self.torque_joints)]
        self._position_multipliers = self._command_multipliers[
            len(self.torque_joints) :
        ]

        link_ids = {self.link(joint_id).name: joint_id for joint_id in range(self.dof)}
        self._finger_links = [link_ids[link_name] for link_name in finger_links]
        self._base_link = link_ids[base_link]

        self._finger_contact_normals = [np.array(n) for n in finger_contact_normals]

        def get_link_mass(joint_id: int):
            return p.getDynamicsInfo(
                self.body_id, joint_id, physicsClientId=self.physics_id
            )[0]

        self._masses = np.array(
            [get_link_mass(joint_id) for joint_id in self.torque_joints]
        )

        # Compute gripper inertia.
        self._inertia = dyn.SpatialInertiad()
        gripper_links = set(
            [self._base_link]
            + self.torque_joints
            + self.position_joints
            + self.finger_links
        )
        for link_id in sorted(gripper_links):
            link = body.Link(physics_id, body_id, link_id)
            T_link_to_ee = T_world_to_ee * link.pose().to_eigen()
            self._inertia += link.inertia * T_link_to_ee

        self.pos_gains = pos_gains
        self.pos_threshold = pos_threshold

        self._gripper_state = GripperState()
        self._q_home = self.get_joint_state(self.joints)[0]
        self.reset()

    @property
    def inertia(self) -> dyn.SpatialInertiad:
        """Gripper inertia."""
        return self._inertia

    @property
    def finger_links(self) -> List[int]:
        """Finger link ids."""
        return self._finger_links

    @property
    def base_link(self) -> int:
        """Base link id."""
        return self._base_link

    @property
    def finger_contact_normals(self) -> List[np.ndarray]:
        """Expected contact normals for each finger link when grasping objects."""
        return self._finger_contact_normals

    def reset(self) -> bool:
        """Removes any grasp constraint and resets the gripper to the open position."""
        self.remove_grasp_constraint()
        self._gripper_state = GripperState()
        self.reset_joints(self._q_home, self.joints)
        self.apply_positions(self._q_home, self.joints)
        return True

    def is_object_grasped(self, body_id: int) -> bool:
        """Detects whether the given body is grasped.

        A body is considered grasped if it is in contact with both finger links
        and the contact normals are pointing inwards.

        Args:
            body_id: Body id for which to check the grasp.
        Returns:
            True if the body is grasped.
        """
        CONTACT_NORMAL_ALIGNMENT = 0.98  # Allows roughly 10 deg error.

        for finger_link_id, finger_contact_normal in zip(
            self.finger_links, self.finger_contact_normals
        ):
            contacts = p.getContactPoints(
                bodyA=body_id,
                bodyB=self.body_id,
                linkIndexB=finger_link_id,
                physicsClientId=self.physics_id,
            )
            if not contacts:
                return False

            T_world_to_finger = self.link(finger_link_id).pose().to_eigen().inverse()
            for contact in contacts:
                # Contact normal on finger, pointing towards obj.
                f_contact = T_world_to_finger.linear @ np.array(contact[7])

                # Make sure contact normal is in the expected direction.
                if f_contact.dot(finger_contact_normal) < CONTACT_NORMAL_ALIGNMENT:
                    return False

                # Contact position on finger.
                pos_contact = (T_world_to_finger * np.append(contact[6], 1.0))[:3]

                # Make sure contact position is on the inner side of the finger.
                if pos_contact.dot(finger_contact_normal) < 0.0:
                    return False

        return True

    def create_grasp_constraint(self, body_id: int, realistic: bool = True) -> bool:
        """Creates a pose constraint to simulate a stable grasp if and only if the body is properly grasped.

        Pybullet is not robust enough to simulate a grasped object using
        force/friction alone, so we create an artificial pose constraint between
        the grasped object and the gripper base.

        Args:
            body_id: Body id to grasp.
            realistic: If false, creates a pose constraint regardless of whether
                the object is in a secure grasp.
        Returns:
            True if the body is successfully grasped.
        """
        self.remove_grasp_constraint()
        if realistic and not self.is_object_grasped(body_id):
            return False

        T_body_to_world = body.Body(self.physics_id, body_id).pose().to_eigen()
        T_ee_to_world = self.link(self._base_link).pose().to_eigen()
        T_body_to_ee = math.Pose.from_eigen(T_ee_to_world.inverse() * T_body_to_world)

        self._gripper_state.grasp_constraint_id = self._create_grasp_constraint(
            body_id, T_body_to_ee
        )
        self._gripper_state.grasp_body_id = body_id
        self._gripper_state.grasp_T_body_to_ee = T_body_to_ee

        return True

    def _create_grasp_constraint(self, body_id: int, T_body_to_ee: math.Pose) -> int:
        return p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=self._base_link,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=np.zeros(3),
            parentFramePosition=T_body_to_ee.pos,
            childFramePosition=np.zeros(3),
            parentFrameOrientation=T_body_to_ee.quat,
            physicsClientId=self.physics_id,
        )

    def remove_grasp_constraint(self) -> None:
        """Removes the grasp constraint if one exists."""
        if self._gripper_state.grasp_constraint_id is None:
            return

        p.removeConstraint(
            self._gripper_state.grasp_constraint_id, physicsClientId=self.physics_id
        )
        self._gripper_state.grasp_constraint_id = None
        self._gripper_state.grasp_body_id = None
        self._gripper_state.grasp_T_body_to_ee = None

    def set_grasp(
        self,
        command: float,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Sets the gripper to the desired grasp (0.0 open, 1.0 closed).

        To actually control the gripper, call `Gripper.update_torques()`.

        Args:
            command: Desired grasp (range from 0.0 open to 1.0 closed).
            pos_gains: kp gains (only used for sim).
            timeout: Uses the timeout specified in the yaml gripper config if None.
        """
        self._gripper_state.command = command
        if timeout is None:
            timeout = self.timeout
        self._pos_gains = self.pos_gains if pos_gains is None else pos_gains

        self._gripper_state.dq_avg = 1.0
        self._gripper_state.iter_timeout = int(timeout / math.PYBULLET_TIMESTEP)

        self._gripper_state.torque_control = True

    def update_torques(self) -> articulated_body.ControlStatus:
        """Computes and applies the torques to control the articulated body to the goal set with `Arm.set_pose_goal().

        Returns:
            Controller status.
        """
        if not self._gripper_state.torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        joint_states = p.getJointStates(
            self.body_id, self._torque_joints, physicsClientId=self.physics_id
        )
        q, dq, _, _ = zip(*joint_states)
        q = np.array(q)
        dq = np.array(dq)
        q_des = self._torque_multipliers * self._gripper_state.command
        q_err = q - q_des

        # Compute commands.
        ddq = -self._pos_gains[0] * q_err - self._pos_gains[1] * dq
        tau = self._masses * ddq
        current = q[0] / self._torque_multipliers[0]
        q_command = self._position_multipliers * current

        self.apply_torques(tau)
        self.apply_positions(q_command)

        self._gripper_state.dq_avg = 0.5 * abs(dq[0]) + 0.5 * self._gripper_state.dq_avg
        self._gripper_state.iter_timeout -= 1

        # Return position converged.
        if q_err.dot(q_err) < self.pos_threshold * self.pos_threshold:
            return articulated_body.ControlStatus.POS_CONVERGED

        # Return velocity converged.
        if self._gripper_state.dq_avg < 0.0001:
            return articulated_body.ControlStatus.VEL_CONVERGED

        # Return timeout.
        if self._gripper_state.iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT

        return articulated_body.ControlStatus.IN_PROGRESS

    def freeze_grasp(self) -> None:
        """Disables torque control and freezes the grasp with position control."""
        self._gripper_state.torque_control = False
        q = self.get_joint_state(self.joints)[0]
        self.apply_positions(q, self.joints)

    def get_state(self) -> Dict[str, Any]:
        state = {
            "articulated_body": super().get_state(),
            "gripper": copy.deepcopy(self._gripper_state),
        }
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state["articulated_body"])
        self._gripper_state = copy.deepcopy(state["gripper"])

        if self._gripper_state.grasp_constraint_id is not None:
            assert self._gripper_state.grasp_body_id is not None
            assert self._gripper_state.grasp_T_body_to_ee is not None

            self._gripper_state.grasp_constraint_id = self._create_grasp_constraint(
                self._gripper_state.grasp_body_id,
                self._gripper_state.grasp_T_body_to_ee,
            )
