import abc
import dataclasses
import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import ctrlutils
import numpy as np

from temporal_policies.envs.pybullet.sim import math


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


def register_object(
    object_name: str,
    redis: ctrlutils.RedisClient,
    graphics: Union[Graphics, Sequence[Graphics]],
    key_pos: Optional[str] = None,
    key_ori: Optional[str] = None,
    key_scale: Optional[str] = None,
    key_matrix: Optional[str] = None,
) -> None:
    """Registers an object with Redis.

    Args:
        object_name: Redis object name.
        redis: Redis client.
        graphics: Graphics object or list of graphics.
        key_pos: Optional position Redis key.
        key_ori: Optional orientation Redis key.
        key_scale: Optioanl scale redis key.
        key_matrix: Optional matrix transformation Redis key.
    """

    if key_pos is None:
        key_pos = ""
    if key_ori is None:
        key_ori = ""
    if key_scale is None:
        key_scale = ""
    if key_matrix is None:
        key_matrix = ""
    if isinstance(graphics, Graphics):
        graphics = [graphics]

    redis.set(
        f"temporal_policies::model::object::{object_name}",  # TODO: Model keys
        json.dumps(
            {
                "graphics": [g.to_dict() for g in graphics],
                "key_pos": key_pos,
                "key_ori": key_ori,
                "key_scale": key_scale,
                "key_matrix": key_matrix,
                "axis_size": 0.1,
            }
        ),
    )
