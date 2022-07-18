from typing import List, Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import articulated_body, math


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
        pos_threshold: Tuple[float, float],
        ori_threshold: Tuple[float, float],
        timeout: float,
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
            pos_threshold: (position, velocity) error threshold for position convergence.
            ori_threshold: (orientation, angular velocity) threshold for orientation convergence.
            timeout: Default command timeout.
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

        self.pos_threshold = np.array(pos_threshold, dtype=np.float64)
        self.ori_threshold = np.array(ori_threshold, dtype=np.float64)

        self._ab = dyn.ArticulatedBody(dyn.urdf.load_model(arm_urdf))
        self.ab.q, self.ab.dq = self.get_state(self.torque_joints)

        self._q_limits = np.array(
            [self.link(link_id).joint_limits for link_id in self.torque_joints]
        ).T

        self._pos_des: Optional[np.ndarray] = None
        self._quat_des: Optional[np.ndarray] = None
        self._torque_control = False

        # TODO: Debugging.
        # self._redis = ctrlutils.RedisClient(port=6000)
        # redisgl.register_object(
        #     "lambda_pos",
        #     self._redis,
        #     redisgl.Graphics("lambda_pos", redisgl.Sphere(0.01)),
        #     key_pos="franka_panda::sensor::pos",
        #     key_ori="",
        #     key_scale="",
        #     key_matrix="franka_panda::opspace::Lambda_pos",
        # )

    @property
    def ab(self) -> dyn.ArticulatedBody:
        """Spatialdyn articulated body."""
        return self._ab

    def reset(self) -> bool:
        """Disables torque control and esets the arm to the home configuration (bypassing simulation)."""
        self._pos_des = None
        self._quat_des = None
        self._torque_control = False
        return self.goto_configuration(self.q_home, skip_simulation=True)

    def set_pose_goal(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[np.ndarray] = None,
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
            self._pos_des = pos
        if quat is not None:
            self._quat_des = quat
        if timeout is None:
            timeout = self.timeout
        self._pos_gains = self.pos_gains if pos_gains is None else pos_gains
        self._ori_gains = self.ori_gains if ori_gains is None else ori_gains

        self._dx_avg = 1.0
        self._w_avg = 1.0
        self._iter_timeout = int(timeout / math.PYBULLET_TIMESTEP)

        self._torque_control = True

    def ee_pose(self) -> math.Pose:
        self.ab.q, self.ab.dq = self.get_state(self.torque_joints)
        T_ee_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        return math.Pose(
            T_ee_to_world.translation, eigen.Quaterniond(T_ee_to_world.linear).coeffs
        )

    def update_torques(self) -> articulated_body.ControlStatus:
        """Computes and applies the torques to control the articulated body to the goal set with `Arm.set_pose_goal().

        Returns:
            Controller status.
        """
        if not self._torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        self.ab.q, self.ab.dq = self.get_state(self.torque_joints)

        if (self._q_limits[0] >= self.ab.q).any() or (
            self.ab.q >= self._q_limits[1]
        ).any():
            return articulated_body.ControlStatus.ABORTED
        # TODO: Debugging.
        # J = dyn.jacobian(self.ab, -1, offset=self.ee_offset)
        # Lambda = dyn.opspace.inertia(self.ab, J, svd_epsilon=0.01)
        # self._redis.set_matrix(
        #     "franka_panda::sensor::pos",
        #     dyn.cartesian_pose(self.ab, -1, offset=np.array([0., 0., 0.107])).translation,
        # )
        # self._redis.set_matrix("franka_panda::opspace::Lambda_pos", Lambda[:3, :3])
        # self._redis.set_matrix("franka_panda::opspace::Lambda_ori", Lambda[3:, 3:])

        # Compute torques.
        tau, status = dyn.opspace_control(
            self.ab,
            pos=self._pos_des,
            ori=self._quat_des,
            joint=self.q_home,
            pos_gains=self._pos_gains,
            ori_gains=self._ori_gains,
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
        self._dx_avg = 0.5 * np.sqrt(dx.dot(dx)) + 0.5 * self._dx_avg
        self._w_avg = 0.5 * np.sqrt(w.dot(w)) + 0.5 * self._w_avg
        if self._dx_avg < 0.001 and self._w_avg < 0.02:
            return articulated_body.ControlStatus.VEL_CONVERGED

        # Return timeout.
        self._iter_timeout -= 1
        if self._iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT

        return articulated_body.ControlStatus.IN_PROGRESS

    def goto_configuration(self, q: np.ndarray, skip_simulation: bool = False) -> bool:
        """Sets the robot to the desired joint configuration.

        Args:
            q: Joint configuration.
            skip_simulation: Whether to forcibly set the joint positions or use
                torque control to achieve them.
        Returns:
            True if the controller converges to the desired position or zero
            velocity, false if the command times out.
        """
        if skip_simulation:
            self._torque_control = False
            self.reset_joints(q, self.torque_joints)
            self.apply_positions(q, self.torque_joints)
            self.ab.q, self.ab.dq = self.get_state(self.torque_joints)
            return True
        else:
            # TODO: Implement torque control.
            raise NotImplementedError
