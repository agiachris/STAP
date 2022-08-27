import abc
import dataclasses
import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import ctrlutils
import numpy as np

from temporal_policies.envs.pybullet.sim import math


KEY_ARGS = "webapp::simulator::args"
KEY_RESOURCES = "webapp::resources::simulator"


class Geometry(abc.ABC):
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class Box(Geometry):
    type = "box"

    def __init__(self, scale: Union[Sequence[float], np.ndarray]):
        self.scale = list(scale)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "scale": self.scale,
        }


class Capsule(Geometry):
    type = "capsule"

    def __init__(self, radius: float, length: float):
        self.radius = radius
        self.length = length

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "radius": self.radius,
            "length": self.length,
        }


class Cylinder(Geometry):
    type = "cylinder"

    def __init__(self, radius: float, length: float):
        self.radius = radius
        self.length = length

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "radius": self.radius,
            "length": self.length,
        }


class Sphere(Geometry):
    type = "sphere"

    def __init__(self, radius: float):
        self.radius = radius

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "radius": self.radius,
        }


class Mesh(Geometry):
    type = "mesh"

    def __init__(self, path: str, scale: Union[Sequence[float], np.ndarray]):
        self.path = path
        self.scale = list(scale)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "mesh": self.path,
            "scale": self.scale,
        }


@dataclasses.dataclass
class Material:
    name: str = ""
    rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rgba": self.rgba,
            "texture": self.texture,
        }


@dataclasses.dataclass
class Graphics:
    name: str
    geometry: Geometry
    material: Material = Material()
    T_to_parent: math.Pose = math.Pose()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "T_to_parent": self.T_to_parent.to_dict(),
            "geometry": self.geometry.to_dict(),
            "material": self.material.to_dict(),
        }


def register_resource_path(redis: ctrlutils.RedisClient, path: str) -> None:
    redis.sadd(KEY_RESOURCES, {path})


def unregister_resource_path(redis: ctrlutils.RedisClient, path: str) -> None:
    redis.srem(KEY_RESOURCES, {path})


@dataclasses.dataclass
class ModelKeys:
    key_namespace: str
    key_robots_prefix: str
    key_objects_prefix: str
    key_trajectories_prefix: str
    key_cameras_prefix: str

    def __init__(self, key_namespace: str):
        self.key_namespace = key_namespace
        self.key_robots_prefix = key_namespace + "::model::robot::"
        self.key_objects_prefix = key_namespace + "::model::object::"
        self.key_trajectories_prefix = key_namespace + "::model::trajectory::"
        self.key_cameras_prefix = key_namespace + "::model::camera::"

    def to_dict(self) -> Dict:
        return {
            "key_robots_prefix": self.key_robots_prefix,
            "key_objects_prefix": self.key_objects_prefix,
            "key_trajectories_prefix": self.key_trajectories_prefix,
            "key_cameras_prefix": self.key_cameras_prefix,
        }


def register_model_keys(redis: ctrlutils.RedisClient, model_keys: ModelKeys) -> None:
    redis.set(
        f"{KEY_ARGS}::{model_keys.key_namespace}", json.dumps(model_keys.to_dict())
    )


def unregister_model_keys(redis: ctrlutils.RedisClient, model_keys: ModelKeys) -> None:
    redis.delete(f"{KEY_ARGS}::{model_keys.key_namespace}")


@dataclasses.dataclass
class ObjectModel:
    name: str
    graphics: Union[Graphics, Sequence[Graphics]]
    key_pos: str = ""
    key_ori: str = ""
    key_scale: str = ""
    key_matrix: str = ""
    axis_size: float = 0.1

    def to_dict(self) -> Dict:
        graphics = (
            [self.graphics] if isinstance(self.graphics, Graphics) else self.graphics
        )
        return {
            "graphics": [g.to_dict() for g in graphics],
            "key_pos": self.key_pos,
            "key_ori": self.key_ori,
            "key_scale": self.key_scale,
            "key_matrix": self.key_matrix,
            "axis_size": self.axis_size,
        }


def register_object(
    redis: ctrlutils.RedisClient,
    model_keys: ModelKeys,
    name: Optional[str] = None,
    graphics: Optional[Union[Graphics, Sequence[Graphics]]] = None,
    key_pos: str = "",
    key_ori: str = "",
    key_scale: str = "",
    key_matrix: str = "",
    axis_size: float = 0.1,
    object: Optional[ObjectModel] = None,
) -> None:
    """Registers an object with Redis.

    Args:
        redis: Redis client.
        model_keys: Redisgl app namespace.
        name: Redis object name.
        graphics: Graphics object or list of graphics.
        key_pos: Optional position Redis key.
        key_ori: Optional orientation Redis key.
        key_scale: Optioanl scale redis key.
        key_matrix: Optional matrix transformation Redis key.
        object_model: Optional object model instead of all of the above optionals.
    """
    if object is None:
        if name is None or graphics is None:
            raise ValueError("name and graphics must be specified if object is None")
        object = ObjectModel(
            name=name,
            graphics=graphics,
            key_pos=key_pos,
            key_ori=key_ori,
            key_scale=key_scale,
            key_matrix=key_matrix,
            axis_size=axis_size,
        )

    redis.set(model_keys.key_objects_prefix + object.name, json.dumps(object.to_dict()))
