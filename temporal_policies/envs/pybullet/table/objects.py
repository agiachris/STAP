import dataclasses
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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
        name: str,
        is_static: bool = False,
    ):
        super().__init__(physics_id, body_id)

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

    def set_state(self, state: object_state.ObjectState) -> None:
        self.set_pose(state.pose())

    def reset(self) -> None:
        pass

    @classmethod
    def create(
        cls,
        physics_id: int,
        object_type: Optional[str],
        object_kwargs: Dict[str, Any] = {},
        object_groups: Dict[str, "ObjectGroup"] = {},
        **kwargs
    ) -> "Object":
        object_class = Null if object_type is None else globals()[object_type]
        if issubclass(object_class, Variant):
            kwargs["object_groups"] = object_groups
        return object_class(physics_id=physics_id, **object_kwargs, **kwargs)

    def isinstance(self, class_or_tuple: type) -> bool:
        return isinstance(self, class_or_tuple)

    @property
    def size(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def shapes(self) -> Sequence[shapes.Shape]:
        return []

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class Urdf(Object):
    AABB_MARGIN = 0.001  # Pybullet seems to expand aabbs by at least this amount.

    def __init__(
        self,
        physics_id: int,
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
            name=name,
            is_static=is_static,
        )

        xyz_min, xyz_max = self.aabb()
        self._size = xyz_max - xyz_min - 2 * Urdf.AABB_MARGIN

    @property
    def size(self) -> np.ndarray:
        return self._size


class Box(Object):
    def __init__(
        self,
        physics_id: int,
        name: str,
        size: Union[List[float], np.ndarray],
        color: Union[List[float], np.ndarray],
        mass: float = 0.1,
    ):
        box = shapes.Box(size=np.array(size), mass=mass, color=np.array(color))
        body_id = shapes.create_body(box, physics_id=physics_id)
        self._shape = box

        super().__init__(
            physics_id=physics_id, body_id=body_id, name=name, is_static=mass == 0.0
        )

        self._state.box_size = box.size

    @property
    def size(self) -> np.ndarray:
        return self._state.box_size

    @property
    def shapes(self) -> Sequence[shapes.Shape]:
        return [self._shape]


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
        self._shapes = [joint, handle, head]
        body_id = shapes.create_body(
            self.shapes, link_parents=[0, 0], physics_id=physics_id
        )

        super().__init__(
            physics_id=physics_id, body_id=body_id, name=name, is_static=mass == 0.0
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

    @property
    def shapes(self) -> Sequence[shapes.Shape]:
        return self._shapes

    # def aabb(self) -> np.ndarray:
    #     raise NotImplementedError


class Rack(Object):
    TOP_THICKNESS = 0.01
    LEG_THICKNESS = 0.01

    def __init__(
        self,
        physics_id: int,
        name: str,
        size: Union[List[float], np.ndarray],
        color: Union[List[float], np.ndarray],
        mass: float = 1.0,
    ):
        mass /= 7  # Divide mass among all 7 parts.
        top = shapes.Box(
            size=np.array([*size[:2], Rack.TOP_THICKNESS]),
            mass=mass,
            color=np.array(color),
            pose=math.Pose(
                pos=np.array([0.0, 0.0, -Rack.TOP_THICKNESS / 2]),
                quat=eigen.Quaterniond.identity().coeffs,
            ),
        )
        xy_legs = np.array([(x, y) for x in (-1, 1) for y in (-1, 1)]) * (
            (np.array(size[:2])[None, :] - Rack.LEG_THICKNESS) / 2
        )
        legs = [
            shapes.Box(
                size=np.array(
                    [
                        Rack.LEG_THICKNESS,
                        Rack.LEG_THICKNESS,
                        size[2] - Rack.TOP_THICKNESS - Rack.LEG_THICKNESS,
                    ]
                ),
                mass=mass,
                color=np.array([0.0, 0.0, 0.0, 1.0]),
                pose=math.Pose(
                    pos=np.array(
                        [
                            *xy_leg,
                            -(size[2] + Rack.TOP_THICKNESS - Rack.LEG_THICKNESS) / 2,
                        ]
                    ),
                    quat=eigen.Quaterniond.identity().coeffs,
                ),
            )
            for xy_leg in xy_legs
        ]
        stabilizers = [
            shapes.Box(
                size=np.array([size[0], Rack.LEG_THICKNESS, Rack.LEG_THICKNESS]),
                mass=mass,
                color=np.array([0.0, 0.0, 0.0, 1.0]),
                pose=math.Pose(
                    pos=np.array([0.0, y_leg, -size[2] + Rack.LEG_THICKNESS / 2]),
                    quat=eigen.Quaterniond.identity().coeffs,
                ),
            )
            for y_leg in xy_legs[:2, 1]
        ]
        self._shapes = [top, *legs, *stabilizers]
        body_id = shapes.create_body(
            self.shapes,
            link_parents=[0] * (len(legs) + len(stabilizers)),
            physics_id=physics_id,
        )

        super().__init__(
            physics_id=physics_id, body_id=body_id, name=name, is_static=mass == 0.0
        )

        self._state.box_size = np.array(size)

    @property
    def size(self) -> np.ndarray:
        return self._state.box_size

    @property
    def shapes(self) -> Sequence[shapes.Shape]:
        return self._shapes


class Null(Object):
    def __init__(self, physics_id: int, name: str):
        sphere = shapes.Sphere(radius=0.001)
        body_id = shapes.create_body(sphere, physics_id=physics_id)

        super().__init__(
            physics_id=physics_id, body_id=body_id, name=name, is_static=True
        )

    def enable_collisions(self) -> None:
        pass

    def unfreeze(self) -> None:
        pass


class WrapperObject(Object):
    def __init__(self, body: Object):
        self.body = body
        self.name = body.name

    def isinstance(self, class_or_tuple: type) -> bool:
        return self.body.isinstance(class_or_tuple)

    # Body methods.

    @property
    def physics_id(self) -> int:  # type: ignore
        return self.body.physics_id

    @property
    def body_id(self) -> int:  # type: ignore
        return self.body.body_id

    @property
    def dof(self) -> int:
        return self.body.dof

    def freeze(self) -> None:
        self.body.freeze()

    def unfreeze(self) -> None:
        self.body.unfreeze()

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

    @property
    def shapes(self) -> Sequence[shapes.Shape]:
        return self.body.shapes

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


class ObjectGroup:
    def __init__(self, physics_id: int, name: str, objects: List[Dict[str, Any]]):
        self._name = name
        self._objects = [
            Object.create(physics_id=physics_id, name=name, **obj_config)
            for obj_config in objects
        ]
        self.reset()

    @property
    def name(self) -> str:
        return self._name

    @property
    def objects(self) -> List[Object]:
        return self._objects

    def reset(self) -> None:
        self.available_indices = list(range(len(self.objects)))

        # Hide objects below table.
        for obj in self.objects:
            obj.disable_collisions()
            obj.set_pose(math.Pose(pos=np.array([0.0, 0.0, -0.5])))
            obj.freeze()

    def pop_index(self, idx: Optional[int] = None) -> int:
        """Pops an index from the list of available indices.

        Args:
            idx: If specified, this index will be popped from the list.

        Returns:
            Popped index.
        """
        if idx is None:
            idx_available = random.randrange(len(self.available_indices))
        else:
            idx_available = self.available_indices.index(idx)

        return self.available_indices.pop(idx_available)

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[Object]:
        return iter(self.objects)

    def __getitem__(self, idx: Union[int, slice]):
        return self.objects[idx]


class Variant(WrapperObject):
    def __init__(
        self,
        physics_id: int,
        name: str,
        variants: Optional[List[Dict[str, Any]]] = None,
        group: Optional[str] = None,
        object_groups: Dict[str, ObjectGroup] = {},
    ):
        self.name = name

        if variants is None and group is None:
            raise ValueError("One of variants or group must be specified")
        elif variants is not None and group is not None:
            raise ValueError("Only one of variants or group can be specified")

        if variants is not None:
            self._variants: Union[List[Object], ObjectGroup] = [
                Object.create(physics_id=self.physics_id, name=self.name, **obj_config)
                for obj_config in variants
            ]
        else:
            assert group is not None
            self._variants = object_groups[group]

        self._body: Optional[Object] = None
        self._idx_variant: Optional[int] = None

    @property
    def body(self) -> Object:  # type: ignore
        if self._body is None:
            raise RuntimeError("Variant.reset() must be called first")
        return self._body

    @property
    def variants(self) -> Union[List[Object], ObjectGroup]:
        return self._variants

    def set_variant(self, idx_variant: Optional[int], lock: bool = False) -> None:
        """Sets the variant for debugging purposes.

        Args:
            idx_variant: Index of the variant to set.
            lock: Whether to lock the variant so it remains the same upon resetting.
        """
        if isinstance(self.variants, ObjectGroup):
            idx_variant = self.variants.pop_index(idx_variant)
        else:
            if idx_variant is None:
                idx_variant = random.randrange(len(self.variants))

            # Hide unused variants below table.
            for i, obj in enumerate(self.variants):
                if i == idx_variant:
                    continue
                obj.disable_collisions()
                obj.set_pose(math.Pose(pos=np.array([0.0, 0.0, -0.5])))
                obj.freeze()

        self._body = self.variants[idx_variant]
        self._idx_variant = idx_variant if lock else None

    def reset(self) -> None:
        self.set_variant(self._idx_variant)
        self.enable_collisions()
        self.unfreeze()
        self.body.reset()
