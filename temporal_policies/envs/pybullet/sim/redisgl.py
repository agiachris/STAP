import dataclasses
import pathlib
from typing import Dict, Optional, Tuple

import ctrlutils
from ctrlutils import eigen
import numpy as np
from redis.exceptions import ConnectionError
import spatialdyn as dyn

from temporal_policies.envs.pybullet.real import redisgl


@dataclasses.dataclass
class RedisKeys:
    namespace: str
    control_pos: str
    control_ori: str
    control_pos_des: str
    control_ori_des: str
    opspace_inertia_pos: str
    opspace_inertia_ori: str
    sensor_q: str
    sensor_dq: str
    sensor_pos: str
    sensor_ori: str


class RedisGl:
    def __init__(
        self,
        redis_host: Optional[str],
        redis_port: Optional[int],
        redis_password: Optional[str],
        redis_keys: Optional[Dict[str, str]],
        arm_urdf: str,
        gripper_offset: Tuple[float, float, float],
        ee_offset: np.ndarray,
    ):
        self._is_active = False
        if redis_host is None:
            return

        if redis_port is None or redis_password is None or redis_keys is None:
            raise ValueError(
                "Redis can only be connected if redis_host, redis_port, "
                "redis_password, and redis_keys are not None"
            )

        self._redis_keys = RedisKeys(**redis_keys)
        self._redis = ctrlutils.RedisClient(redis_host, redis_port, redis_password)

        # Only publish to redisgl if robot controller isn't already running.
        try:
            redis_robot_controller = self._redis.get(
                f"{redisgl.KEY_ARGS}::{self._redis_keys.namespace}"
            )
        except ConnectionError:
            return
        if redis_robot_controller is not None:
            return

        self._redis_pipe = self._redis.pipeline()

        self._ab = dyn.ArticulatedBody(dyn.urdf.load_model(arm_urdf))
        self.gripper_offset = np.array(gripper_offset)
        self.ee_offset = ee_offset

        # Register app.
        self._model_keys = redisgl.ModelKeys(self._redis_keys.namespace)
        self._arm_urdf_path = str(pathlib.Path(arm_urdf).parent.absolute())
        redisgl.register_resource_path(self._redis_pipe, self._arm_urdf_path)
        redisgl.register_model_keys(self._redis_pipe, self._model_keys)
        self._redis_pipe.execute()

        self._is_active = True

        # Register robot.
        if self._redis.get(self._model_keys.key_robots_prefix + self.ab.name) is None:
            redisgl.register_robot(
                self._redis_pipe,
                self._model_keys,
                redisgl.RobotModel(
                    articulated_body=self.ab, key_q=self._redis_keys.sensor_q
                ),
            )

        # Register goal pose.
        redisgl.register_object(
            self._redis,
            self._model_keys,
            redisgl.ObjectModel(
                name="pose_des",
                graphics=redisgl.Graphics("pose_des", redisgl.Sphere(0.01)),
                key_pos=self._redis_keys.control_pos_des,
                key_ori=self._redis_keys.control_ori_des,
            ),
        )

        # Register opspace inertia.
        redisgl.register_object(
            self._redis,
            self._model_keys,
            redisgl.ObjectModel(
                name="lambda_pos",
                graphics=redisgl.Graphics(
                    "lambda_pos",
                    redisgl.Sphere(0.01),
                    material=redisgl.Material(rgba=(1.0, 1.0, 1.0, 0.5)),
                ),
                key_pos=self._redis_keys.control_pos,
                key_matrix=self._redis_keys.opspace_inertia_pos,
                axis_size=0.01,
            ),
        )

    @property
    def ab(self) -> dyn.ArticulatedBody:
        return self._ab

    def __del__(self) -> None:
        if not self._is_active:
            return

        redisgl.unregister_resource_path(self._redis_pipe, self._arm_urdf_path)
        redisgl.unregister_model_keys(self._redis_pipe, self._model_keys)
        self._redis_pipe.execute()

    def update(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        pos_des: Optional[np.ndarray],
        quat_des: Optional[np.ndarray],
    ) -> None:
        if not self._is_active:
            return

        self.ab.q, self.ab.dq = q, dq

        # Compute ee pose.
        T_ee_to_world = dyn.cartesian_pose(self.ab, -1, self.ee_offset)
        quat_ee_to_world = eigen.Quaterniond(T_ee_to_world.linear)

        # Compute opspace inertia.
        J = dyn.jacobian(self.ab, -1, offset=self.ee_offset)
        Lambda = dyn.opspace.inertia(self.ab, J, svd_epsilon=0.01)

        # Get sensor pos for robotiq alignment.
        sensor_pos = (
            quat_ee_to_world * (self.gripper_offset - self.ee_offset)
            + T_ee_to_world.translation
        )

        self._redis_pipe.set_matrix(self._redis_keys.sensor_q, q)
        self._redis_pipe.set_matrix(self._redis_keys.sensor_dq, dq)
        self._redis_pipe.set_matrix(self._redis_keys.sensor_pos, sensor_pos)
        self._redis_pipe.set_matrix(
            self._redis_keys.sensor_ori, quat_ee_to_world.coeffs
        )
        self._redis_pipe.set_matrix(
            self._redis_keys.control_pos, T_ee_to_world.translation
        )
        self._redis_pipe.set_matrix(
            self._redis_keys.control_ori, quat_ee_to_world.coeffs
        )
        self._redis_pipe.set_matrix(
            self._redis_keys.opspace_inertia_pos, Lambda[:3, :3]
        )
        self._redis_pipe.set_matrix(
            self._redis_keys.opspace_inertia_ori, Lambda[3:, 3:]
        )
        if pos_des is not None:
            self._redis_pipe.set_matrix(self._redis_keys.control_pos_des, pos_des)
        if quat_des is not None:
            self._redis_pipe.set_matrix(self._redis_keys.control_ori_des, quat_des)

        self._redis_pipe.execute()
