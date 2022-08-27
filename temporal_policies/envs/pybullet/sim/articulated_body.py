import abc
import copy
import dataclasses
import enum
from typing import Any, Dict, List, Optional, Tuple

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


@dataclasses.dataclass
class ArticulatedBodyState:
    positions: np.ndarray
    torques: np.ndarray
    torque_mode: bool = False


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

        joint_ids = {get_joint_name(joint_id): joint_id for joint_id in range(self.dof)}
        self._torque_joints = [joint_ids[joint] for joint in torque_joints]
        self._position_joints = [joint_ids[joint] for joint in position_joints]

        len_joints = max(self.joints) + 1
        self._articulated_body_state = ArticulatedBodyState(
            np.zeros(len_joints), np.zeros(len_joints)
        )

        self.timeout = timeout

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
        return self.torque_joints + self.position_joints

    def link(self, link_id: int) -> body.Link:
        """Link with the given id."""
        return body.Link(self.physics_id, self.body_id, link_id)

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
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
        if not self._articulated_body_state.torque_mode:
            p.setJointMotorControlArray(
                self.body_id,
                joints,
                p.POSITION_CONTROL,
                forces=np.zeros_like(torques),
                physicsClientId=self.physics_id,
            )
            p.setJointMotorControlArray(
                self.body_id,
                joints,
                p.VELOCITY_CONTROL,
                forces=np.zeros_like(torques),
                physicsClientId=self.physics_id,
            )
            if set_torque_mode:
                self._articulated_body_state.torque_mode = True
            self._articulated_body_state.positions[joints] = float("nan")

        # Apply torques.
        p.setJointMotorControlArray(
            self.body_id,
            joints,
            p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.physics_id,
        )
        self._articulated_body_state.torques[joints] = torques

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
            self._articulated_body_state.torque_mode = False

        p.setJointMotorControlArray(
            self.body_id,
            joints,
            p.TORQUE_CONTROL,
            forces=np.zeros_like(q),
            physicsClientId=self.physics_id,
        )
        p.setJointMotorControlArray(
            self.body_id,
            joints,
            p.POSITION_CONTROL,
            targetPositions=q,
            physicsClientId=self.physics_id,
        )
        self._articulated_body_state.torques[joints] = 0
        self._articulated_body_state.positions[joints] = q

    def get_state(self) -> Dict[str, Any]:
        return {"articulated_body": copy.deepcopy(self._articulated_body_state)}

    def set_state(self, state: Dict[str, Any]) -> None:
        self._articulated_body_state = copy.deepcopy(state["articulated_body"])

        idx_disabled_position = np.isnan(
            self._articulated_body_state.positions[self.joints]
        )
        idx_torque = self._articulated_body_state.torques[self.joints] > 0

        joints = np.array(self.joints)
        disabled_joints = joints[idx_disabled_position]
        torque_joints = joints[idx_torque]
        position_joints = joints[~idx_disabled_position]

        null_command = np.zeros(len(disabled_joints))
        p.setJointMotorControlArray(
            self.body_id,
            disabled_joints,
            p.POSITION_CONTROL,
            forces=null_command,
            physicsClientId=self.physics_id,
        )
        p.setJointMotorControlArray(
            self.body_id,
            disabled_joints,
            p.VELOCITY_CONTROL,
            forces=null_command,
            physicsClientId=self.physics_id,
        )
        p.setJointMotorControlArray(
            self.body_id,
            torque_joints,
            p.TORQUE_CONTROL,
            forces=self._articulated_body_state.torques[torque_joints],
            physicsClientId=self.physics_id,
        )
        p.setJointMotorControlArray(
            self.body_id,
            position_joints,
            p.POSITION_CONTROL,
            targetPositions=self._articulated_body_state.positions[position_joints],
            physicsClientId=self.physics_id,
        )
