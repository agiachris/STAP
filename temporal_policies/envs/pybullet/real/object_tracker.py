import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Union

import ctrlutils
from ctrlutils import eigen
import numpy as np

from temporal_policies.envs.pybullet.real import redisgl
from temporal_policies.envs.pybullet.sim import math, shapes
from temporal_policies.envs.pybullet.table.objects import Object, Variant


def create_pose(shape: shapes.Shape) -> redisgl.Pose:
    if shape.pose is None:
        return redisgl.Pose()
    elif isinstance(shape, shapes.Cylinder):
        quat_pybullet_to_redisgl = eigen.Quaterniond(
            eigen.AngleAxisd(np.pi / 2, np.array([1.0, 0.0, 0.0]))
        )
        quat = eigen.Quaterniond(shape.pose.quat) * quat_pybullet_to_redisgl
        return redisgl.Pose(shape.pose.pos, quat.coeffs)
    else:
        return redisgl.Pose(shape.pose.pos, shape.pose.quat)


def create_geometry(shape: shapes.Shape) -> redisgl.Geometry:
    if isinstance(shape, shapes.Box):
        return redisgl.Box(scale=shape.size)
    elif isinstance(shape, shapes.Cylinder):
        return redisgl.Cylinder(radius=shape.radius, length=shape.length)
    elif isinstance(shape, shapes.Sphere):
        return redisgl.Sphere(radius=shape.radius)

    raise NotImplementedError(f"Shape type {shape} is not supported.")


def create_graphics(object: Object) -> Sequence[redisgl.Graphics]:
    if isinstance(object, Variant):
        return []

    return [
        redisgl.Graphics(
            name=object.name,
            geometry=create_geometry(shape),
            T_to_parent=create_pose(shape),
        )
        for shape in object.shapes
    ]


def create_object_model(object: Object, key_namespace: str) -> redisgl.ObjectModel:
    return redisgl.ObjectModel(
        name=object.name,
        graphics=create_graphics(object),
        key_pos=f"{key_namespace}::objects::{object.name}::pos",
        key_ori=f"{key_namespace}::objects::{object.name}::ori",
    )


class ObjectTracker:
    def __init__(
        self,
        objects: Dict[str, Object],
        redis_host: str,
        redis_port: int,
        redis_password: str,
        key_namespace: str,
        object_key_prefix: str,
        assets_path: Union[str, pathlib.Path],
    ):
        self._redis = ctrlutils.RedisClient(redis_host, redis_port, redis_password)
        self._redis_pipe = self._redis.pipeline()
        self._object_key_prefix = object_key_prefix

        self._assets_path = str(pathlib.Path(assets_path).absolute())
        redisgl.register_resource_path(self._redis_pipe, self._assets_path)
        self._model_keys = redisgl.ModelKeys(key_namespace)
        redisgl.register_model_keys(self._redis_pipe, self._model_keys)

        self._redis_pipe.execute()
        self._tracked_objects = []  # self.get_tracked_objects(objects.values())
        for object in objects.values():
            try:
                redisgl.register_object(
                    self._redis_pipe,
                    self._model_keys,
                    object=create_object_model(object, key_namespace),
                )
            except NotImplementedError:
                continue
            self._tracked_objects.append(object)
        self._redis_pipe.execute()

    def __del__(self) -> None:
        redisgl.unregister_resource_path(self._redis_pipe, self._assets_path)
        redisgl.unregister_model_keys(self._redis_pipe, self._model_keys)
        for object in self._tracked_objects:
            redisgl.unregister_object(self._redis_pipe, self._model_keys, object.name)
        self._redis_pipe.execute()

    def get_tracked_objects(self, objects: Iterable[Object]) -> List[Object]:
        for object in objects:
            self._redis_pipe.get(self._object_key_prefix + object.name + "::pos")
        object_models = self._redis_pipe.execute()

        return [
            object
            for object, object_model in zip(objects, object_models)
            if object_model is not None
        ]

    def update_poses(
        self,
        objects: Optional[Iterable[Object]] = None,
        exclude: Optional[Sequence[Object]] = None,
    ) -> List[Object]:
        if objects is None:
            objects = self._tracked_objects

        # Query all object poses.
        for object in objects:
            self._redis_pipe.get(self._object_key_prefix + object.name + "::pos")
            self._redis_pipe.get(self._object_key_prefix + object.name + "::ori")
        b_object_poses = self._redis_pipe.execute()

        # Set returned poses.
        updated_objects = []
        for i, object in enumerate(objects):
            if exclude is not None and object in exclude:
                continue
            b_object_pos = b_object_poses[2 * i]
            b_object_quat = b_object_poses[2 * i + 1]
            if b_object_pos is None or b_object_quat is None:
                continue

            object_pos = ctrlutils.redis.decode_matlab(b_object_pos)
            object_quat = ctrlutils.redis.decode_matlab(b_object_quat)

            object.set_pose(math.Pose(object_pos, object_quat))
            updated_objects.append(object)

        return updated_objects

    def send_poses(self, objects: Optional[Iterable[Object]] = None) -> None:
        if objects is None:
            objects = self._tracked_objects

        for object in objects:
            pose = object.pose()
            self._redis_pipe.set_matrix(
                self._object_key_prefix + object.name + "::pos", pose.pos
            )
            self._redis_pipe.set_matrix(
                self._object_key_prefix + object.name + "::ori", pose.quat
            )
        self._redis_pipe.execute()
