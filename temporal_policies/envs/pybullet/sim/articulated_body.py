import abc
import enum
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

from temporal_policies.envs.pybullet.sim import body


class ControlStatus(enum.Enum):
    IN_PROGRESS = 0
    POS_CONVERGED = 1
    VEL_CONVERGED = 2
    TIMEOUT = 3
    ABORTED = 4
    UNINITIALIZED = 5


class ArticulatedBody(body.Body, abc.ABC):
    """Wrapper class for controllable articulated bodies in Pybullet."""

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        torque_joints: List[str],
        position_joints: List[str],
        timeout: float,
    ):
        """Constructs the wrapper class.

        Args:
            physics_id: Pybullet physics client id.
            body_id: Pybullet body id.
            torque_joints: List of torque-controlled joint names.
            position_joints: List of position-controlled joint names.
            timeout: Default command timeout.
        """
        super().__init__(physics_id, body_id)

        def get_joint_name(joint_id: int) -> str:
            return p.getJointInfo(
                self.body_id, joint_id, physicsClientId=self.physics_id
            )[1].decode("utf8")

        self._dof = p.getNumJoints(self.body_id, physicsClientId=self.physics_id)
        joint_ids = {get_joint_name(joint_id): joint_id for joint_id in range(self.dof)}
        self._torque_joints = [joint_ids[joint] for joint in torque_joints]
        self._position_joints = [joint_ids[joint] for joint in position_joints]
        self._torque_mode = False

        self.timeout = float(timeout)

    @property
    def torque_joints(self) -> List[int]:
        """List of torque-controlled joint ids."""
        return self._torque_joints

    @property
    def position_joints(self) -> List[int]:
        """List of position-controlled joint ids."""
        return self._position_joints

    @property
    def joints(self) -> List[int]:
        """List of torque and position-controlled joint ids."""
        return self._torque_joints + self._position_joints

    @property
    def dof(self) -> int:
        """Total number of joints in the articulated body, including non-controlled joints."""
        return self._dof

    def link(self, link_id: int) -> body.Link:
        """Link with the given id."""
        return body.Link(self.physics_id, self.body_id, link_id)

    def get_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        joint_states = p.getJointStates(
            self.body_id, joints, physicsClientId=self.physics_id
        )
        q, dq, _, _ = zip(*joint_states)
        return np.array(q), np.array(dq)

    def reset_joints(self, q: np.ndarray, joints: List[int]) -> None:
        """Resets the positions of the given joints and sets their velocity to 0.

        Args:
            joints: List of joint ids.
        """
        for i, q_i in zip(joints, q):
            p.resetJointState(self.body_id, i, q_i, 0, physicsClientId=self.physics_id)

    @abc.abstractmethod
    def update_torques(self) -> ControlStatus:
        """Computes and applies the torques to control the articulated body to the previously set goal.

        Returns:
            Controller status.
        """
        raise NotImplementedError

    def apply_torques(
        self, torques: np.ndarray, joints: Optional[List[int]] = None
    ) -> None:
        """Applies torques to the given joints.

        Pybullet requires disabling position and velocity control in order to
        use torque control. To prevent this overhead every time torques are
        applied, this method checks if this articulated body is already in
        torque mode, but only if the `joints` arg is None and the default
        `self.torque_joints` are used.

        Args:
            torques: Desired torques.
            joints: Optional list of joints. If None, `self.torque_joints` are used.
        """
        if joints is None:
            joints = self.torque_joints
            set_torque_mode = True
        else:
            set_torque_mode = False

        # Disable position/velocity control.
        if not self._torque_mode:
            null_command = [0] * len(joints)
            p.setJointMotorControlArray(
                self.body_id,
                joints,
                p.POSITION_CONTROL,
                forces=null_command,
                physicsClientId=self.physics_id,
            )
            p.setJointMotorControlArray(
                self.body_id,
                joints,
                p.VELOCITY_CONTROL,
                forces=null_command,
                physicsClientId=self.physics_id,
            )
            if set_torque_mode:
                self._torque_mode = True

        # Apply torques.
        p.setJointMotorControlArray(
            self.body_id,
            joints,
            p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.physics_id,
        )

    def apply_positions(
        self, q: np.ndarray, joints: Optional[List[int]] = None
    ) -> None:
        """Sets the joints to the desired positions.

        This method will disable torque mode.

        Args:
            q: Desired positions.
            joints: Optional list of joints. If None, `self.position_joints` are used.
        """

        if joints is None:
            joints = self.position_joints
        else:
            self._torque_mode = False

        p.setJointMotorControlArray(
            self.body_id,
            joints,
            p.POSITION_CONTROL,
            targetPositions=q,
            physicsClientId=self.physics_id,
        )
