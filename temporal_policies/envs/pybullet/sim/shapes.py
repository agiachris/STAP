import dataclasses
import enum
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pybullet as p

from temporal_policies.envs.pybullet.sim.math import Pose


def create_body(
    shapes: Union["Shape", Sequence["Shape"]],
    link_parents: Optional[Sequence[int]] = None,
    physics_id: int = 0,
) -> int:
    if isinstance(shapes, Shape):
        base_shape = shapes
        link_shapes: Sequence[Shape] = []
    else:
        base_shape = shapes[0]
        link_shapes = shapes[1:]

    base_collision_id, base_visual_id = base_shape.create_visual(
        physics_id, is_base=True
    )
    kwargs: Dict[str, Any] = {
        "baseMass": base_shape.mass,
        "baseCollisionShapeIndex": base_collision_id,
        "baseVisualShapeIndex": base_visual_id,
    }
    if base_shape.pose is not None:
        kwargs["baseInertialFramePosition"] = base_shape.pose.pos
        kwargs["baseInertialFrameOrientation"] = base_shape.pose.quat

    if len(link_shapes) > 0:
        masses, poses, joints, collision_ids, visual_ids = zip(
            *[
                (shape.mass, shape.pose, shape.joint, *shape.create_visual(physics_id))
                for shape in link_shapes
            ]
        )

        kwargs["linkMasses"] = masses
        kwargs["linkCollisionShapeIndices"] = collision_ids
        kwargs["linkVisualShapeIndices"] = visual_ids

        link_poses = [Pose() if pose is None else pose for pose in poses]
        link_inertia_poses = [Pose()] * len(poses)
        kwargs["linkPositions"] = [pose.pos for pose in link_poses]
        kwargs["linkOrientations"] = [pose.quat for pose in link_poses]
        kwargs["linkInertialFramePositions"] = [pose.pos for pose in link_inertia_poses]
        kwargs["linkInertialFrameOrientations"] = [
            pose.quat for pose in link_inertia_poses
        ]

        link_joints = [Joint() if joint is None else joint for joint in joints]
        if link_parents is None:
            link_parents = list(range(len(joints)))
        kwargs["linkParentIndices"] = link_parents
        kwargs["linkJointTypes"] = [int(joint.joint_type) for joint in link_joints]
        kwargs["linkJointAxis"] = [joint.axis for joint in link_joints]

    body_id = p.createMultiBody(
        physicsClientId=physics_id,
        **kwargs,
    )

    return body_id


class JointType(enum.IntEnum):
    REVOLUTE = p.JOINT_REVOLUTE
    PRISMATIC = p.JOINT_PRISMATIC
    SPHERICAL = p.JOINT_SPHERICAL
    FIXED = p.JOINT_FIXED


@dataclasses.dataclass
class Joint:
    joint_type: JointType = JointType.FIXED
    axis: np.ndarray = np.array([0.0, 0.0, 1.0])


@dataclasses.dataclass
class Shape:
    mass: float = 0.0
    color: Optional[np.ndarray] = None
    pose: Optional[Pose] = None
    joint: Optional[Joint] = None

    def visual_kwargs(
        self, is_base: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        collision_kwargs = {}
        visual_kwargs = {}
        if self.color is not None:
            visual_kwargs["rgbaColor"] = self.color

        if is_base and self.pose is not None:
            collision_kwargs["collisionFramePosition"] = self.pose.pos
            collision_kwargs["collisionFrameOrientation"] = self.pose.quat
            visual_kwargs["visualFramePosition"] = self.pose.pos
            visual_kwargs["visualFrameOrientation"] = self.pose.quat

        return collision_kwargs, visual_kwargs

    def create_visual(self, physics_id: int, is_base: bool = False) -> Tuple[int, int]:
        collision_kwargs, visual_kwargs = self.visual_kwargs(is_base)

        collision_id = p.createCollisionShape(
            physicsClientId=physics_id, **collision_kwargs
        )
        visual_id = p.createVisualShape(physicsClientId=physics_id, **visual_kwargs)

        return collision_id, visual_id


@dataclasses.dataclass
class Box(Shape):
    size: np.ndarray = np.array([0.1, 0.1, 0.1])

    def visual_kwargs(
        self, is_base: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        collision_kwargs, visual_kwargs = super().visual_kwargs(is_base)

        collision_kwargs["shapeType"] = p.GEOM_BOX
        collision_kwargs["halfExtents"] = self.size / 2

        visual_kwargs["shapeType"] = p.GEOM_BOX
        visual_kwargs["halfExtents"] = self.size / 2

        return collision_kwargs, visual_kwargs


@dataclasses.dataclass
class Cylinder(Shape):
    radius: float = 0.05
    length: float = 0.1

    def visual_kwargs(
        self, is_base: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        collision_kwargs, visual_kwargs = super().visual_kwargs(is_base)

        collision_kwargs["shapeType"] = p.GEOM_CYLINDER
        collision_kwargs["radius"] = self.radius
        collision_kwargs["height"] = self.length

        visual_kwargs["shapeType"] = p.GEOM_CYLINDER
        visual_kwargs["radius"] = self.radius
        visual_kwargs["length"] = self.length

        return collision_kwargs, visual_kwargs


@dataclasses.dataclass
class Sphere(Shape):
    radius: float = 0.05

    def visual_kwargs(
        self, is_base: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        collision_kwargs, visual_kwargs = super().visual_kwargs(is_base)

        collision_kwargs["shapeType"] = p.GEOM_SPHERE
        collision_kwargs["radius"] = self.radius

        visual_kwargs["shapeType"] = p.GEOM_SPHERE
        visual_kwargs["radius"] = self.radius

        return collision_kwargs, visual_kwargs
