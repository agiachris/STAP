from typing import List, Optional, Tuple, Union

import numpy as np
import spatialdyn as dyn
# import ctrlutils  # TODO: debug

from temporal_policies.envs.pybullet.sim import articulated_body, math, redisgl


class Arm(articulated_body.ArticulatedBody):
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
        joint_gains: Tuple[float, float],
        pos_threshold: Tuple[float, float],
        ori_threshold: Tuple[float, float],
        timeout: float,
    ):
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
        self.joint_gains = np.array(joint_gains, dtype=np.float64)

        self.pos_threshold = np.array(pos_threshold, dtype=np.float64)
        self.ori_threshold = np.array(ori_threshold, dtype=np.float64)

        self._ab = dyn.ArticulatedBody(dyn.urdf.load_model(arm_urdf))

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
        return self._ab

    # def ee_pose(self) -> Pose:
    #     return Pose.from_eigen(dyn.cartesian_pose(self.ab, offset=self.ee_offset))

    def reset(self) -> bool:
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

    def update_torques(self) -> articulated_body.ControlStatus:
        if not self._torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        self.ab.q, self.ab.dq = self.get_state(self.torque_joints)

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
            joint_gains=self.joint_gains,
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
        if skip_simulation:
            self._torque_control = False
            self.reset_joints(q, self.torque_joints)
            self.apply_positions(q, self.torque_joints)
            self.ab.q, self.ab.dq = self.get_state(self.torque_joints)
            return True
        else:
            raise NotImplementedError
