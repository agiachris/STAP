import dataclasses
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import body, math, shapes
from temporal_policies.envs.pybullet.table import object_state


@dataclasses.dataclass
class Object(body.Body):
    name: str
    is_static: bool = False

    def __init__(
        self,
        physics_id: int,
        body_id: int,
        idx_object: int,
        name: str,
        is_static: bool = False,
    ):
        super().__init__(physics_id, body_id)

        self.idx_object = idx_object
        self.name = name
        self.is_static = is_static

        T_pybullet_to_obj = super().pose().to_eigen()
        self._modified_axes = not T_pybullet_to_obj.is_approx(
            eigen.Isometry3d.identity()
        )
        if self._modified_axes:
            self._T_pybullet_to_obj = T_pybullet_to_obj
            self._T_obj_to_pybullet = T_pybullet_to_obj.inverse()

        self._state = object_state.ObjectState()

    def pose(self) -> math.Pose:
        if not self._modified_axes:
            return super().pose()

        return math.Pose.from_eigen(super().pose().to_eigen() * self._T_obj_to_pybullet)

    def set_pose(self, pose: math.Pose) -> None:
        if not self._modified_axes:
            return super().set_pose(pose)

        return super().set_pose(
            math.Pose.from_eigen(pose.to_eigen() * self._T_pybullet_to_obj)
        )

    def disable_collisions(self) -> None:
        for link_id in range(self.dof):
            p.setCollisionFilterGroupMask(
                self.body_id, link_id, 0, 0, physicsClientId=self.physics_id
            )

    def enable_collisions(self) -> None:
        for link_id in range(self.dof):
            p.setCollisionFilterGroupMask(
                self.body_id, link_id, 1, 0xFF, physicsClientId=self.physics_id
            )

    @property
    def inertia(self) -> dyn.SpatialInertiad:
        try:
            return self._obj_inertia  # type: ignore
        except AttributeError:
            pass

        self._obj_inertia = super().inertia
        if self._modified_axes:
            self._obj_inertia = self._obj_inertia * self._T_pybullet_to_obj

        T_world_to_obj = self.pose().to_eigen().inverse()
        for link_id in range(self.dof):
            link = body.Link(self.physics_id, self.body_id, link_id)
            T_link_to_obj = T_world_to_obj * link.pose().to_eigen()
            self._obj_inertia += link.inertia * T_link_to_obj

        return self._obj_inertia

    def state(self) -> object_state.ObjectState:
        pose = self.pose()
        aa = eigen.AngleAxisd(eigen.Quaterniond(pose.quat))
        self._state.pos = pose.pos
        self._state.aa = aa.angle * aa.axis

        return self._state

    def reset(self) -> None:
        pass

    @classmethod
    def create(
        cls,
        physics_id: int,
        idx_object: int,
        object_type: str,
        object_kwargs: Dict[str, Any],
        **kwargs
    ) -> "Object":
        object_class = globals()[object_type]
        return object_class(
            physics_id=physics_id, idx_object=idx_object, **object_kwargs, **kwargs
        )

    def isinstance(self, class_or_tuple: type) -> bool:
        return isinstance(self, class_or_tuple)

    @property
    def size(self) -> np.ndarray:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class Urdf(Object):
    def __init__(
        self,
        physics_id: int,
        idx_object: int,
        name: str,
        path: str,
        is_static: bool = False,
    ):
        body_id = p.loadURDF(
            fileName=path,
            useFixedBase=is_static,
            physicsClientId=physics_id,
        )

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            idx_object=idx_object,
            name=name,
            is_static=is_static,
        )

        # xyz_min, xyz_max = self.aabb()
        # self._size = xyz_max - xyz_min

    @property
    def size(self) -> np.ndarray:
        raise NotImplementedError
        return self._size


class Box(Object):
    def __init__(
        self,
        physics_id: int,
        idx_object: int,
        name: str,
        size: Union[List[float], np.ndarray],
        color: Union[List[float], np.ndarray],
        mass: float = 0.1,
    ):
        box = shapes.Box(size=np.array(size), mass=mass, color=np.array(color))
        body_id = shapes.create_body(box, physics_id=physics_id)

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            idx_object=idx_object,
            name=name,
            is_static=mass == 0.0,
        )

        self._state.box_size = box.size

    @property
    def size(self) -> np.ndarray:
        return self._state.box_size


class Hook(Object):
    @staticmethod
    def compute_link_positions(
        head_length: float, handle_length: float, handle_y: float, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dy = (
            0.5
            * np.sign(handle_y)
            * max(0, (abs(handle_y) - 1.0) * head_length / 2 + radius)
        )
        pos_handle = np.array([-radius / 2, handle_y * head_length / 2 - dy, 0.0])
        pos_head = np.array([(handle_length - radius) / 2, -dy, 0.0])
        pos_joint = np.array(
            [(handle_length - radius) / 2, handle_y * head_length / 2 - dy, 0.0]
        )

        return pos_handle, pos_head, pos_joint

    def __init__(
        self,
        physics_id: int,
        idx_object: int,
        name: str,
        head_length: float,
        handle_length: float,
        handle_y: float,
        color: Union[List[float], np.ndarray],
        radius: float = 0.02,
        mass: float = 0.1,
    ):
        if not isinstance(color, np.ndarray):
            color = np.array(color)

        pos_handle, pos_head, pos_joint = Hook.compute_link_positions(
            head_length=head_length,
            handle_length=handle_length,
            handle_y=handle_y,
            radius=radius,
        )
        handle = shapes.Cylinder(
            radius=radius,
            length=handle_length,
            mass=(handle_length / (head_length + handle_length + radius)) * mass,
            color=color,
            pose=math.Pose(
                pos=pos_handle,
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
                pos=pos_head,
                quat=eigen.Quaterniond(
                    eigen.AngleAxisd(angle=np.pi / 2, axis=np.array([1.0, 0.0, 0.0]))
                ).coeffs,
            ),
        )
        joint = shapes.Sphere(
            radius=radius,
            mass=(radius / (head_length + handle_length + radius)) * mass,
            color=color,
            pose=math.Pose(pos=pos_joint),
        )
        body_id = shapes.create_body(
            [joint, handle, head], link_parents=[0, 0], physics_id=physics_id
        )

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            idx_object=idx_object,
            name=name,
            is_static=mass == 0.0,
        )

        self._state.head_length = head_length
        self._state.handle_length = handle_length
        self._state.handle_y = handle_y
        self._radius = radius

        self._size = np.array(
            [handle_length + radius, head_length + 2 * abs(pos_head[1]), 2 * radius]
        )
        self._bbox = np.array([-self.size / 2, self.size / 2])

    @property
    def head_length(self) -> float:
        return self._state.head_length  # type: ignore

    @property
    def handle_length(self) -> float:
        return self._state.handle_length  # type: ignore

    @property
    def handle_y(self) -> float:
        return self._state.handle_y  # type: ignore

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

    # def aabb(self) -> np.ndarray:
    #     raise NotImplementedError


class Null(Object):
    def __init__(
        self,
        physics_id: int,
        idx_object: int,
        name: str,
    ):
        sphere = shapes.Sphere(radius=0.001)
        body_id = shapes.create_body(sphere, physics_id=physics_id)

        super().__init__(
            physics_id=physics_id,
            body_id=body_id,
            idx_object=idx_object,
            name=name,
            is_static=sphere.mass == 0,
        )


class Variant(Object):
    def __init__(
        self,
        physics_id: int,
        idx_object: int,
        name: str,
        variants: List[Dict[str, Any]],
    ):
        self.physics_id = physics_id
        self.idx_object = idx_object
        self.name = name

        self._variants = [
            Object.create(
                physics_id=self.physics_id,
                idx_object=idx_object,
                name=self.name,
                **obj_kwargs,
            )
            for obj_kwargs in variants
        ]
        self._masses = [
            p.getDynamicsInfo(obj.body_id, -1, physicsClientId=self.physics_id)[0]
            for obj in self.variants
        ]
        self._body: Optional[Object] = None

    @property
    def body(self) -> Object:
        if self._body is None:
            raise RuntimeError("Variant.reset() must be called first")
        return self._body

    @property
    def variants(self) -> List[Object]:
        return self._variants

    def set_variant(self, idx_body: int) -> None:
        for i, body in enumerate(self.variants):
            if i == idx_body:
                continue
            body.disable_collisions()
            body.set_pose(math.Pose(pos=np.array([0.0, 0.0, -0.5])))
            body.freeze()
        self._body = self.variants[idx_body]

    def reset(self) -> None:
        self.set_variant(idx_body=random.randrange(len(self.variants)))
        self.body.enable_collisions()
        self.body.unfreeze()
        self.body.reset()

    def isinstance(self, class_or_tuple: type) -> bool:
        return self.body.isinstance(class_or_tuple)

    # Body methods.

    @property
    def body_id(self) -> int:  # type: ignore
        return self.body.body_id

    @property
    def dof(self) -> int:
        return self.body.dof

    # Object methods.

    def pose(self) -> math.Pose:
        return self.body.pose()

    def set_pose(self, pose: math.Pose) -> None:
        self.body.set_pose(pose)

    @property
    def inertia(self) -> dyn.SpatialInertiad:
        return self.body.inertia

    def state(self) -> object_state.ObjectState:
        return self.body.state()

    @property
    def is_static(self) -> bool:  # type: ignore
        return self.body.is_static

    @property
    def size(self) -> np.ndarray:
        return self.body.size

    # Hook methods.

    @property
    def head_length(self) -> float:
        assert isinstance(self.body, Hook)
        return self.body.head_length

    @property
    def handle_length(self) -> float:
        assert isinstance(self.body, Hook)
        return self.body.handle_length

    @property
    def handle_y(self) -> float:
        assert isinstance(self.body, Hook)
        return self.body.handle_y

    @property
    def radius(self) -> float:
        assert isinstance(self.body, Hook)
        return self.body.radius

    @property
    def bbox(self) -> np.ndarray:
        assert isinstance(self.body, Hook)
        return self.body.bbox
