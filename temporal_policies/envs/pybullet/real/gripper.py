import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import ctrlutils
from ctrlutils import eigen
import numpy as np
import pybullet as p

from temporal_policies.envs.pybullet.sim import articulated_body, gripper as sim_gripper


@dataclasses.dataclass
class RedisKeys:
    control_pub_command: str
    control_pub_status: str
    sensor_pos: str


class Gripper(sim_gripper.Gripper):
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
        redis_host: str,
        redis_port: int,
        redis_password: str,
        redis_keys: Dict[str, str],
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
            redis_host: Gripper redis host (usually localhost).
            redis_port: Gripper redis port.
            redis_password: Gripper redis password.
            redis_keys: Gripper redis keys.
        """
        self._redis = ctrlutils.RedisClient(redis_host, redis_port, redis_password)
        self._redis_pipe = self._redis.pipeline()
        self._redis_sub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._redis_keys = RedisKeys(**redis_keys)

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            T_world_to_ee=T_world_to_ee,
            torque_joints=torque_joints,
            position_joints=position_joints,
            finger_links=finger_links,
            base_link=base_link,
            command_multipliers=command_multipliers,
            finger_contact_normals=finger_contact_normals,
            inertia_kwargs=inertia_kwargs,
            pos_gains=pos_gains,
            pos_threshold=pos_threshold,
            timeout=timeout,
        )

        for link_id in range(self.dof):
            p.setCollisionFilterGroupMask(
                self.body_id, link_id, 0, 0, physicsClientId=self.physics_id
            )

    def get_joint_state(self, joints: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the position and velocities of the given joints.

        Gets the joint state from the real gripper via Redis and applies it to pybullet.

        Args:
            joints: List of joint ids.
        Returns:
            Joint positions and velocities (q, dq).
        """
        if joints != self.joints:
            raise NotImplementedError

        b_gripper_pos = self._redis.get(self._redis_keys.sensor_pos)
        if b_gripper_pos is None:
            raise RuntimeError("Unable to get Redis key:", self._redis_keys.sensor_pos)
        gripper_pos = float(b_gripper_pos.decode("utf8")) / 255

        q = self._command_multipliers * gripper_pos

        # Update pybullet joints.
        self.apply_positions(q, joints)

        return q, np.zeros_like(q)

    def reset_joints(self, q: np.ndarray, joints: List[int]) -> None:
        raise NotImplementedError

    def apply_torques(
        self, torques: np.ndarray, joints: Optional[List[int]] = None
    ) -> None:
        raise NotImplementedError

    def reset(self) -> bool:
        """Removes any grasp constraint and resets the gripper to the open position."""
        self.remove_grasp_constraint()
        self._gripper_state = sim_gripper.GripperState()
        self.set_grasp(0)
        while self.update_torques() == articulated_body.ControlStatus.IN_PROGRESS:
            continue

        return True

    def is_object_grasped(self, body_id: int) -> bool:
        """Detects whether the given body is grasped.

        A body is considered grasped if the gripper is perfectly closed (sim) or
        not mostly closed (real).

        Args:
            body_id: Body id for which to check the grasp.
        Returns:
            True if the body is grasped.
        """
        # Assume object is grasped if gripper is not fully closed (255).
        gripper_pos = int(self._redis.get(self._redis_keys.sensor_pos).decode("utf8"))
        return gripper_pos == 255 or gripper_pos < 250

    def set_grasp(
        self,
        command: float,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().set_grasp(command, pos_gains, timeout)
        self._gripper_state.torque_control = False

        self._redis_sub.subscribe(self._redis_keys.control_pub_status)

        robotiq_command = int(command * 255 + 0.5)
        self._redis.publish(self._redis_keys.control_pub_command, robotiq_command)

    def update_torques(self) -> articulated_body.ControlStatus:
        """Gets the latest status from the Redis gripper controller.

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
        q, dq = self.get_joint_state(self.joints)
        self.apply_positions(q, self.joints)

        # Return in progress.
        if message is None:
            return articulated_body.ControlStatus.IN_PROGRESS

        # TODO: Timeout

        return articulated_body.ControlStatus.VEL_CONVERGED

    def set_state(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError
