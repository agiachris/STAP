import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import articulated_body, math, redisgl


@dataclasses.dataclass
class ArmState:
    """Mutable arm state."""

    pos_des: Optional[np.ndarray] = None
    quat_des: Optional[np.ndarray] = None
    pos_gains: Union[Tuple[float, float], np.ndarray] = (64.0, 16.0)
    ori_gains: Union[Tuple[float, float], np.ndarray] = (64.0, 16.0)
    torque_control: bool = False
    dx_avg: float = 0.0
    w_avg: float = 0.0
    iter_timeout: int = 0


class Arm(articulated_body.ArticulatedBody):
    """Arm controlled with operational space control."""

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        arm_urdf: str,
        torque_joints: List[str],
        q_home: List[float],
        ee_offset: Tuple[float, float, float],
        pos_gains: Tuple[float, float],
        ori_gains: Tuple[float, float],
        nullspace_joint_gains: Tuple[float, float],
        nullspace_joint_indices: List[int],
        pos_threshold: Tuple[float, float],
        ori_threshold: Tuple[float, float],
        timeout: float,
        redisgl_config: Optional[Dict[str, Any]] = None,
    ):
        """Constructs the arm from yaml config.

        Args:
            physics_id: Pybullet physics client id.
            body_id: Pybullet body id.
            arm_urdf: Path to arm-only urdf for spatialdyn. This urdf will be
                used for computing opspace commands.
            torque_joints: List of torque-controlled joint names.
            q_home: Home joint configuration.
            ee_offset: Position offset from last link com to end-effector operational point.
            pos_gains: (kp, kv) position gains.
            ori_gains: (kp, kv) orientation gains.
            nullspace_joint_gains: (kp, kv) nullspace joint gains.
            nullspace_joint_indices: Joints to control in the nullspace.
            pos_threshold: (position, velocity) error threshold for position convergence.
            ori_threshold: (orientation, angular velocity) threshold for orientation convergence.
            timeout: Default command timeout.
            redisgl_config: Config for setting up RedisGl visualization.
        """
        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            torque_joints=torque_joints,
            position_joints=[],
            timeout=timeout,
        )

        self.q_home = np.array(q_home, dtype=np.float64)
        self.ee_offset = np.array(ee_offset, dtype=np.float64)

        self.pos_gains = np.array(pos_gains, dtype=np.float64)
        self.ori_gains = np.array(ori_gains, dtype=np.float64)
        self.nullspace_joint_gains = np.array(nullspace_joint_gains, dtype=np.float64)
        self.nullspace_joint_indices = list(nullspace_joint_indices)

        self.pos_threshold = np.array(pos_threshold, dtype=np.float64)
        self.ori_threshold = np.array(ori_threshold, dtype=np.float64)

        self._ab = dyn.ArticulatedBody(dyn.urdf.load_model(arm_urdf))
        self.ab.q = self.q_home
        T_home_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        self.quat_home = eigen.Quaterniond(T_home_to_world.linear)
        self.home_pose = self.ee_pose(update=False)

        self._q_limits = np.array(
            [self.link(link_id).joint_limits for link_id in self.torque_joints]
        ).T

        self._redisgl = (
            None
            if redisgl_config is None
            else redisgl.RedisGl(
                **redisgl_config, ee_offset=self.ee_offset, arm_urdf=arm_urdf
            )
        )

        self._arm_state = ArmState()
        self.reset()

    @property
    def ab(self) -> dyn.ArticulatedBody:
        """Spatialdyn articulated body."""
        return self._ab

    def reset(self) -> bool:
        """Disables torque control and resets the arm to the home configuration (bypassing simulation)."""
        self._arm_state = ArmState()
        self.set_configuration_goal(self.q_home, skip_simulation=True)
        return True

    def set_pose_goal(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Sets the pose goal.

        To actually control the robot, call `Arm.update_torques()`.

        Args:
            pos: Optional position. Maintains current position if None.
            quat: Optional quaternion. Maintains current orientation if None.
            pos_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            ori_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            timeout: Uses the timeout specified in the yaml arm config if None.
        """
        if pos is not None:
            self._arm_state.pos_des = pos
        if quat is not None:
            if isinstance(quat, np.ndarray):
                quat = eigen.Quaterniond(quat)
            quat = quat * self.quat_home.inverse()
            self._arm_state.quat_des = quat.coeffs
        if timeout is None:
            timeout = self.timeout
        self._arm_state.pos_gains = self.pos_gains if pos_gains is None else pos_gains
        self._arm_state.ori_gains = self.ori_gains if ori_gains is None else ori_gains

        self._arm_state.dx_avg = 1.0
        self._arm_state.w_avg = 1.0
        self._arm_state.iter_timeout = int(timeout / math.PYBULLET_TIMESTEP)
        self._arm_state.torque_control = True

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        q, dq = super().get_joint_state(joints)
        if self._redisgl is not None:
            self._redisgl.update(
                q, dq, self._arm_state.pos_des, self._arm_state.quat_des
            )
        return q, dq

    def set_configuration_goal(
        self, q: np.ndarray, skip_simulation: bool = False
    ) -> None:
        """Sets the robot to the desired joint configuration.

        Args:
            q: Joint configuration.
            skip_simulation: Whether to forcibly set the joint positions or use
                torque control to achieve them.
        """
        if skip_simulation:
            self._arm_state.torque_control = False
            self.reset_joints(q, self.torque_joints)
            self.apply_positions(q, self.torque_joints)
            self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
            return

        # TODO: Implement torque control.
        raise NotImplementedError

    def ee_pose(self, update: bool = True) -> math.Pose:
        if update:
            self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
        T_ee_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        quat_ee_to_world = eigen.Quaterniond(T_ee_to_world.linear)
        quat_ee = quat_ee_to_world * self.quat_home.inverse()
        return math.Pose(T_ee_to_world.translation, quat_ee.coeffs)

    def update_torques(self) -> articulated_body.ControlStatus:
        """Computes and applies the torques to control the articulated body to the goal set with `Arm.set_pose_goal().

        Returns:
            Controller status.
        """
        if not self._arm_state.torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

        if (self._q_limits[0] >= self.ab.q).any() or (
            self.ab.q >= self._q_limits[1]
        ).any():
            return articulated_body.ControlStatus.ABORTED

        # Compute torques.
        q_nullspace = np.array(self.ab.q)
        if self._arm_state.pos_des is not None:
            # Assume base joint rotates about z-axis.
            q_nullspace[0] = np.arctan2(
                self._arm_state.pos_des[1], self._arm_state.pos_des[0]
            )
        q_nullspace[self.nullspace_joint_indices] = self.q_home[
            self.nullspace_joint_indices
        ]
        tau, status = dyn.opspace_control(
            self.ab,
            pos=self._arm_state.pos_des,
            ori=self._arm_state.quat_des,
            joint=q_nullspace,
            pos_gains=self._arm_state.pos_gains,
            ori_gains=self._arm_state.ori_gains,
            joint_gains=self.nullspace_joint_gains,
            task_pos=self.ee_offset,
            pos_threshold=self.pos_threshold,
            ori_threshold=self.ori_threshold,
        )

        # Abort if singular.
        if status >= 16:
            return articulated_body.ControlStatus.ABORTED

        self.apply_torques(tau)

        # Return positioned converged.
        if status == 15:
            return articulated_body.ControlStatus.POS_CONVERGED

        # Return velocity converged.
        dx_w = dyn.jacobian(self.ab, offset=self.ee_offset).dot(self.ab.dq)
        dx = dx_w[:3]
        w = dx_w[3:]

        self._arm_state.dx_avg = (
            0.5 * np.sqrt(dx.dot(dx)) + 0.5 * self._arm_state.dx_avg
        )
        self._arm_state.w_avg = 0.5 * np.sqrt(w.dot(w)) + 0.5 * self._arm_state.w_avg
        if self._arm_state.dx_avg < 0.001 and self._arm_state.w_avg < 0.02:
            return articulated_body.ControlStatus.VEL_CONVERGED

        # Return timeout.
        self._arm_state.iter_timeout -= 1
        if self._arm_state.iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT

        return articulated_body.ControlStatus.IN_PROGRESS

    def get_state(self) -> Dict[str, Any]:
        state = {
            "articulated_body": super().get_state(),
            "arm": copy.deepcopy(self._arm_state),
        }
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state["articulated_body"])
        self._arm_state = copy.deepcopy(state["arm"])
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)
