import dataclasses
from typing import Dict, List, Optional, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import symbolic

from temporal_policies.envs.pybullet.sim import body, math, shapes
from temporal_policies.envs.pybullet.table import object_state


@dataclasses.dataclass
class Object(body.Body):
    name: str
    is_static: bool
    initial_state: Optional[str]

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        name: str,
        is_static: bool = False,
        initial_state: Optional[str] = None,
    ):
        super().__init__(physics_id, body_id)

        self.name = name
        self.is_static = is_static
        self.initial_state = initial_state

        T_pybullet_to_obj = super().pose().to_eigen()
        self._modified_axes = not T_pybullet_to_obj.is_approx(eigen.Isometry3d.identity())
        if self._modified_axes:
            self._T_pybullet_to_obj = T_pybullet_to_obj
            self._T_obj_to_pybullet = T_pybullet_to_obj.inverse()

        self._state = object_state.ObjectState()

    def pose(self) -> math.Pose:
        if not self._modified_axes:
            return super().pose()

        return math.Pose.from_eigen(
            super().pose().to_eigen() * self._T_obj_to_pybullet
        )

    def set_pose(self, pose: math.Pose) -> None:
        if not self._modified_axes:
            return super().set_pose(pose)

        return super().set_pose(
            math.Pose.from_eigen(pose.to_eigen() * self._T_pybullet_to_obj)
        )

    def state(self) -> object_state.ObjectState:
        pose = self.pose()
        aa = eigen.AngleAxisd(eigen.Quaterniond(pose.quat))
        self._state.pos = pose.pos
        self._state.aa = aa.angle * aa.axis

        return self._state

    def reset(self, objects: Dict[str, "Object"]) -> None:
        if self.is_static or self.initial_state is None:
            return

        predicate, args = symbolic.parse_proposition(self.initial_state)
        if predicate == "on":
            parent_obj = objects[args[1]]
            xyz_min, xyz_max = parent_obj.aabb()
            xyz = np.zeros(3)
            xyz[:2] = np.random.uniform(0.9 * xyz_min[:2], 0.9 * xyz_max[:2])
            xyz[2] = xyz_max[2] + 0.2
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
            pose = math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)
            self.set_pose(pose)


    @classmethod
    def create(cls, physics_id: int, **kwargs) -> "Object":
        if "urdf" in kwargs:
            return Urdf(physics_id, **kwargs["urdf"])
        elif "box" in kwargs:
            return Box(physics_id, **kwargs["box"])
        elif "hook" in kwargs:
            return Hook(physics_id, **kwargs["hook"])
        else:
            raise NotImplementedError


class Urdf(Object):
    def __init__(
        self,
        physics_id: int,
        name: str,
        path: str,
        is_static: bool = False,
        initial_state: Optional[str] = None,
    ):
        body_id = p.loadURDF(
            fileName=path,
            useFixedBase=is_static,
            physicsClientId=physics_id,
        )

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            name=name,
            is_static=is_static,
            initial_state=initial_state,
        )


class Box(Object):
    def __init__(
        self,
        physics_id: int,
        name: str,
        size: Union[List[float], np.ndarray],
        color: Union[List[float], np.ndarray],
        mass: float = 0.1,
        initial_state: Optional[str] = None,
    ):
        box = shapes.Box(size=np.array(size), mass=mass, color=np.array(color))
        body_id = shapes.create_body(box, physics_id=physics_id)

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            name=name,
            is_static=mass == 0.0,
            initial_state=initial_state,
        )

        self._state.box_size = box.size

    @property
    def size(self) -> np.ndarray:
        return self._state.box_size


@dataclasses.dataclass
class Hook(Object):
    def __init__(
        self,
        physics_id: int,
        name: str,
        head_length: float,
        handle_length: float,
        handle_y: float,
        color: Union[List[float], np.ndarray],
        radius: float = 0.02,
        mass: float = 0.1,
        initial_state: Optional[str] = None,
    ):
        if not isinstance(color, np.ndarray):
            color = np.array(color)

        dy = (
            0.5
            * np.sign(handle_y)
            * max(0, (abs(handle_y) - 1.0) * head_length / 2 + radius)
        )
        handle = shapes.Cylinder(
            radius=radius,
            length=handle_length,
            mass=(handle_length / (head_length + handle_length + radius)) * mass,
            color=color,
            pose=math.Pose(
                pos=np.array([0.0, handle_y * head_length / 2 - dy, 0.0]),
                quat=eigen.Quaterniond(
                    eigen.AngleAxisd(angle=np.pi / 2, axis=np.array([0.0, 1.0, 0.0]))
                ).coeffs,
            ),
        )
        head = shapes.Cylinder(
            radius=radius,
            length=head_length,
            mass=(head_length / (head_length + handle_length + radius)) * mass,
            color=color,
            pose=math.Pose(
                pos=np.array([handle_length / 2, -dy, 0.0]),
                quat=eigen.Quaterniond(
                    eigen.AngleAxisd(angle=np.pi / 2, axis=np.array([1.0, 0.0, 0.0]))
                ).coeffs,
            ),
        )
        joint = shapes.Sphere(
            radius=radius,
            mass=(radius / (head_length + handle_length + radius)) * mass,
            color=color,
            pose=math.Pose(
                pos=np.array([handle_length / 2, handle_y * head_length / 2 - dy, 0.0])
            ),
        )
        body_id = shapes.create_body(
            [joint, handle, head], link_parents=[0, 0], physics_id=physics_id
        )

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            name=name,
            is_static=mass == 0.0,
            initial_state=initial_state,
        )

        self._state.head_length = head_length
        self._state.handle_length = handle_length
        self._state.handle_y = handle_y

    @property
    def head_length(self) -> float:
        return self._state.head_length  # type: ignore

    @property
    def handle_length(self) -> float:
        return self._state.handle_length  # type: ignore

    @property
    def handle_y(self) -> float:
        return self._state.handle_y  # type: ignore
