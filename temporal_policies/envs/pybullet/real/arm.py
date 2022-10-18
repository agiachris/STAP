import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import ctrlutils
from ctrlutils import eigen
import numpy as np
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import arm as sim_arm, articulated_body, math


@dataclasses.dataclass
class RedisKeys:
    control_mode: str
    control_pub_command: str
    control_pub_status: str
    driver_status: str
    sensor_q: str
    sensor_dq: str
    sensor_ori: str


class Arm(sim_arm.Arm):
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
        redis_host: str,
        redis_port: int,
        redis_password: str,
        redis_keys: Dict[str, str],
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
            redis_host: Robot redis host (usually the NUC).
            redis_port: Robot redis port.
            redis_password: Robot redis password.
            redis_keys: Robot redis keys.
        """
        self._redis = ctrlutils.RedisClient(redis_host, redis_port, redis_password)
        self._redis_pipe = self._redis.pipeline()
        self._redis_sub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._redis_keys = RedisKeys(**redis_keys)

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            arm_urdf=arm_urdf,
            torque_joints=torque_joints,
            q_home=q_home,
            ee_offset=ee_offset,
            pos_gains=pos_gains,
            ori_gains=ori_gains,
            nullspace_joint_gains=nullspace_joint_gains,
            nullspace_joint_indices=nullspace_joint_indices,
            pos_threshold=pos_threshold,
            ori_threshold=ori_threshold,
            timeout=timeout,
        )

        self._is_real_world = (
            self._redis.get(self._redis_keys.driver_status) is not None
        )

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Gets the joint state from the real robot via Redis and applies it to pybullet.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        if joints != self.torque_joints:
            raise NotImplementedError

        self._redis_pipe.get(self._redis_keys.sensor_q)
        self._redis_pipe.get(self._redis_keys.sensor_dq)
        b_q, b_dq = self._redis_pipe.execute()
        if b_q is None:
            raise RuntimeError("Unable to get Redis key:", self._redis_keys.sensor_q)
        if b_dq is None:
            raise RuntimeError("Unable to get Redis key:", self._redis_keys.sensor_dq)
        q = ctrlutils.redis.decode_matlab(b_q)
        dq = ctrlutils.redis.decode_matlab(b_dq)

        # Update pybullet joints.
        self.apply_positions(q, joints)

        return q, dq

    def reset_joints(self, q: np.ndarray, joints: List[int]) -> None:
        raise NotImplementedError

    def apply_torques(
        self, torques: np.ndarray, joints: Optional[List[int]] = None
    ) -> None:
        raise NotImplementedError

    def reset(self) -> bool:
        """Resets the pose goals."""
        self._arm_state = sim_arm.ArmState()
        return True

    def set_pose_goal(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().set_pose_goal(pos, quat, pos_gains, ori_gains, timeout)

        pub_command = {
            "type": "pose",
            "pos_tolerance": self.pos_threshold[0],
            "ori_tolerance": self.ori_threshold[0],
            "timeout": self._arm_state.iter_timeout * math.PYBULLET_TIMESTEP,
        }
        if self._arm_state.pos_des is not None:
            pub_command["pos"] = self._arm_state.pos_des.tolist()
        if self._arm_state.quat_des is not None:
            quat_des = eigen.Quaterniond(self._arm_state.quat_des)
            quat_curr = eigen.Quaterniond(
                self._redis.get_matrix(self._redis_keys.sensor_ori)
            )
            quat_des = ctrlutils.near_quaternion(quat_des, quat_curr)
            pub_command["quat"] = quat_des.coeffs.tolist()

        self._redis_sub.subscribe(self._redis_keys.control_pub_status)
        self._redis.publish(
            self._redis_keys.control_pub_command, json.dumps(pub_command)
        )

    def set_configuration_goal(
        self, q: np.ndarray, skip_simulation: bool = False
    ) -> None:
        """Sets the robot to the desired joint configuration.

        Joint space control is not implemented yet, so sets an equivalent pose
        goal instead.

        Args:
            q: Joint configuration.
            skip_simulation: Ignored for the real robot.
        """
        if skip_simulation:
            super().reset_joints(q, self.torque_joints)

        self.ab.q = q
        T_ee_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        quat_ee_to_world = eigen.Quaterniond(T_ee_to_world.linear)
        quat_ee = quat_ee_to_world * self.quat_home.inverse()

        self.set_pose_goal(T_ee_to_world.translation, quat_ee)

    def update_torques(self) -> articulated_body.ControlStatus:
        """Gets the latest status from the Redis opspace controller.

        Returns:
            Controller status.
        """
        message = self._redis_sub.get_message()
        while message is not None:
            if message["data"].decode("utf8") == "done":
                self._redis_sub.unsubscribe(self._redis_keys.control_pub_status)
                break
            message = self._redis_sub.get_message()

        # Update sim.
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

        # Return in progress.
        if message is None:
            return articulated_body.ControlStatus.IN_PROGRESS

        if self._is_real_world:
            control_mode = self._redis.get(self._redis_keys.control_mode).decode("utf8")
            if control_mode == "floating":
                return articulated_body.ControlStatus.ABORTED

        # Return positioned converged.
        T_ee_to_world = dyn.cartesian_pose(self.ab, offset=self.ee_offset)
        quat_ee_to_world = eigen.Quaterniond(T_ee_to_world.linear)
        quat_ee = quat_ee_to_world * self.quat_home.inverse()
        if (
            np.linalg.norm(T_ee_to_world.translation - self._arm_state.pos_des)
            < self.pos_threshold[0]
            and np.linalg.norm(
                ctrlutils.orientation_error(
                    quat_ee, eigen.Quaterniond(self._arm_state.quat_des)
                )
            )
            < self.ori_threshold[0]
        ):
            return articulated_body.ControlStatus.POS_CONVERGED

        # Return velocity converged.
        J = dyn.jacobian(self.ab, -1, offset=self.ee_offset)
        ee_twist = J @ self.ab.dq
        if (
            np.linalg.norm(ee_twist[:3]) < self.pos_threshold[1]
            and np.linalg.norm(ee_twist[3:]) < self.ori_threshold[1]
        ):
            return articulated_body.ControlStatus.VEL_CONVERGED

        # Return timeout.
        return articulated_body.ControlStatus.TIMEOUT

    def set_state(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError
