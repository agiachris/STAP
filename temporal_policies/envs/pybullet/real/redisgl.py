import abc
import dataclasses
import json
from typing import Any, Dict, Sequence, Tuple, Union

import ctrlutils
import numpy as np
import spatialdyn as dyn


KEY_ARGS = "webapp::simulator::args"
KEY_RESOURCES = "webapp::resources::simulator"


@dataclasses.dataclass
class Pose:
    """6d pose.

    Args:
        pos: 3d position.
        quat: xyzw quaternion.
    """

    pos: np.ndarray = np.zeros(3)
    quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    def from_dict(self, pose: Dict[str, Any]) -> "Pose":
        """Creates a pose from dict format."""
        return Pose(
            np.array(pose["pose"]),
            np.array(
                [pose["ori"]["x"], pose["ori"]["y"], pose["ori"]["z"], pose["ori"]["w"]]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts a pose to dict format."""
        return {
            "pos": self.pos.tolist(),
            "ori": {
                "x": self.quat[0],
                "y": self.quat[1],
                "z": self.quat[2],
                "w": self.quat[3],
            },
        }


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
    T_to_parent: Pose = Pose()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "T_to_parent": self.T_to_parent.to_dict(),
            "geometry": self.geometry.to_dict(),
            "material": self.material.to_dict(),
        }


def register_resource_path(redis: ctrlutils.RedisClient, path: str) -> None:
    redis.sadd(KEY_RESOURCES, path)


def unregister_resource_path(redis: ctrlutils.RedisClient, path: str) -> None:
    redis.srem(KEY_RESOURCES, path)


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

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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
    redis: ctrlutils.RedisClient, model_keys: ModelKeys, object: ObjectModel
) -> None:
    """Registers an object with Redis.

    Args:
        redis: Redis client.
        model_keys: Redisgl app namespace.
        object_model: Object model.
    """
    redis.set(model_keys.key_objects_prefix + object.name, json.dumps(object.to_dict()))


def unregister_object(
    redis: ctrlutils.RedisClient, model_keys: ModelKeys, name: str
) -> None:
    """Unregisters an object with Redis.

    Args:
        redis: Redis client.
        name: Object name.espace.
        object_model: Object model.
    """
    redis.delete(model_keys.key_objects_prefix + name)


@dataclasses.dataclass
class RobotModel:
    articulated_body: dyn.ArticulatedBody
    key_q: str
    key_pos: str = ""
    key_ori: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "articulated_body": json.loads(str(self.articulated_body)),
            "key_q": self.key_q,
            "key_pos": self.key_pos,
            "key_ori": self.key_ori,
        }


def register_robot(
    redis: ctrlutils.RedisClient, model_keys: ModelKeys, robot: RobotModel
) -> None:
    redis.set(
        model_keys.key_robots_prefix + robot.articulated_body.name,
        json.dumps(robot.to_dict()),
    )
