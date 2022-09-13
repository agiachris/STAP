import dataclasses
import itertools
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import body, math, shapes
from temporal_policies.envs.pybullet.table import object_state


OBJECT_HIERARCHY = ["rack", "table", "hook", "box"]


def compute_bbox_vertices(
    bbox: np.ndarray, pose: Optional[math.Pose] = None, project_2d: bool = False
) -> np.ndarray:
    """Computes the vertices of the given 3D bounding box.

    Args:
        bbox: Array of shape [2, 3] (min/max, x/y/z).
        pose: Optional pose to transform the vertices.
        project_2d: Whether to return 2D vertices or 3D vertices.

    Returns:
        Array of shape [6, 3] for 3D or [4, 2] for 2D.
    """
    xs, ys, zs = bbox.T

    if project_2d:
        # 2D box with vertices in clockwise order.
        vertices = np.array(
            [[xs[0], ys[0]], [xs[0], ys[1]], [xs[1], ys[1]], [xs[1], ys[0]]]
        )
        if pose is not None:
            vertices = np.concatenate(
                (vertices, np.tile([zs.mean(), 1.0], (vertices.shape[0], 1))), axis=1
            )
            vertices = (vertices @ pose.to_eigen().matrix.T)[:, :2]
    else:
        # 3D box.
        vertices = np.array(list(itertools.product(xs, ys, zs, [1.0])))
        if pose is not None:
            vertices = vertices @ pose.to_eigen().matrix.T
        vertices = vertices[:, :3]

    return vertices


@dataclasses.dataclass
class Object(body.Body):
    name: str
    is_static: bool = False

    def __init__(
        self, physics_id: int, body_id: int, name: str, is_static: bool = False
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

    def reset(self, action_skeleton: List) -> None:
        pass

    @classmethod
    def create(
        cls,
        physics_id: int,
        object_type: Optional[str],
        object_kwargs: Dict[str, Any] = {},
        object_groups: Dict[str, "ObjectGroup"] = {},
        **kwargs,
    ) -> "Object":
        object_class = Null if object_type is None else globals()[object_type]
        if issubclass(object_class, Variant):
            kwargs["object_groups"] = object_groups
        object_kwargs = object_kwargs.copy()
        object_kwargs.update(kwargs)
        return object_class(physics_id=physics_id, **object_kwargs)

    def isinstance(self, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
        return isinstance(self, class_or_tuple)

    def type(self) -> Type["Object"]:
        return type(self)

    @property
    def size(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def bbox(self) -> np.ndarray:
        """Returns the bounding box in the object frame.

        If the origin of the object is at its geometric center, this will be
        equivalent to `(-0.5 * self.size, 0.5 * self.size)`.

        Returns:
            An array of shape [2, 3] (min/max, x/y/z).
        """
        raise NotImplementedError

    def convex_hulls(
        self, world_frame: bool = True, project_2d: bool = False
    ) -> List[np.ndarray]:
        """Computes the object's convex hull.

        These hulls will be used for rough collision checking. By default,
        the vertices will be the 6 corners of the object's bounding box
        (`Object.bbox`).

        Args:
            world_frame: Whether to transform the vertices in world frame or
                leave them in object frame.
            project_2d: Whether to return the 2d convex hull.

        Returns:
            List of arrays of shape [_, 3] or [_, 2], where each array is a
            convex hull.
        """
        pose = self.pose() if world_frame else None
        vertices = compute_bbox_vertices(self.bbox, pose, project_2d)

        return [vertices]

    def aabb(self) -> np.ndarray:
        """Computes the axis-aligned bounding box from the object pose and size.

        This should be more accurate than `super().aabb()`, which gets the aabb
        from Pybullet. Pybullet returns an *enlarged* aabb for the object *base*
        link, while this returns the exact aabb for the entire object.

        Returns:
            An array of shape [2, 3] (min/max, x/y/z).
        """
        vertices = np.concatenate(self.convex_hulls(world_frame=True), axis=0)
        xyz_min = vertices.min(axis=0)
        xyz_max = vertices.max(axis=0)

        return np.array([xyz_min, xyz_max])

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

    def __init__(self, physics_id: int, name: str, path: str, is_static: bool = False):
        body_id = p.loadURDF(
            fileName=path, useFixedBase=is_static, physicsClientId=physics_id
        )

        super().__init__(
            physics_id=physics_id, body_id=body_id, name=name, is_static=is_static
        )

        xyz_min, xyz_max = body.Body.aabb(self)
        xyz_min += Urdf.AABB_MARGIN
        xyz_max -= Urdf.AABB_MARGIN
        self._size = xyz_max - xyz_min
        self._bbox = np.array([xyz_min, xyz_max])

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox


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
        self._bbox = np.array([-0.5 * self.size, 0.5 * self.size])

    @property
    def size(self) -> np.ndarray:
        return self._state.box_size

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

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
        self._bbox = np.array([-0.5 * self.size, 0.5 * self.size])

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

    def convex_hulls(
        self, world_frame: bool = True, project_2d: bool = False
    ) -> List[np.ndarray]:
        """Computes the convex hulls of the handle and head links."""
        handle_pose = self.shapes[1].pose
        head_pose = self.shapes[2].pose
        assert handle_pose is not None and head_pose is not None

        positions = np.array(
            [
                [0.0, handle_pose.pos[1], 0.0],
                [head_pose.pos[0], 0.0, 0.0],
            ]
        )
        sizes = np.array(
            [
                [self.size[0], 2 * self.radius, 2 * self.radius],
                [2 * self.radius, self.size[1], 2 * self.radius],
            ]
        )
        bboxes = np.array([positions - 0.5 * sizes, positions + 0.5 * sizes]).swapaxes(
            0, 1
        )

        pose = self.pose() if world_frame else None
        vertices = [compute_bbox_vertices(bbox, pose, project_2d) for bbox in bboxes]

        return vertices

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
        self._bbox = np.array([-0.5 * self.size, 0.5 * self.size])
        self._bbox[0, 2] = -size[2]
        self._bbox[1, 2] = 0

    @property
    def size(self) -> np.ndarray:
        return self._state.box_size

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

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

    def state(self) -> object_state.ObjectState:
        # Null object state is a zero vector.
        return self._state

    def enable_collisions(self) -> None:
        pass

    def unfreeze(self) -> bool:
        return False


class WrapperObject(Object):
    def __init__(self, body: Object):
        self.body = body
        self.name = body.name

    def isinstance(self, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
        return self.body.isinstance(class_or_tuple)

    def type(self) -> Type["Object"]:
        return type(self.body)

    # Body methods.

    @property
    def body_id(self) -> int:  # type: ignore
        return self.body.body_id

    @property
    def dof(self) -> int:
        return self.body.dof

    def freeze(self) -> bool:
        return self.body.freeze()

    def unfreeze(self) -> bool:
        return self.body.unfreeze()

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
    def bbox(self) -> np.ndarray:
        return self.body.bbox

    def convex_hulls(
        self, world_frame: bool = True, project_2d: bool = False
    ) -> List[np.ndarray]:
        return self.body.convex_hulls(world_frame, project_2d)

    def aabb(self) -> np.ndarray:
        return self.body.aabb()

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


class ObjectGroup:
    def __init__(self, physics_id: int, name: str, objects: List[Dict[str, Any]]):
        from temporal_policies.envs.pybullet.table.utils import load_config

        self._name = name
        self._objects = [
            Object.create(physics_id=physics_id, name=name, **load_config(obj_config))
            for obj_config in objects
        ]

        self._real_indices = [
            i for i, obj in enumerate(self.objects) if not isinstance(obj, Null)
        ]
        self._null_indices = [
            i for i, obj in enumerate(self.objects) if isinstance(obj, Null)
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    def objects(self) -> List[Object]:
        return self._objects

    @property
    def num_real_objects(self) -> int:
        """Number of non-null objects."""
        return len(self._real_indices)

    @property
    def num_null_objects(self) -> int:
        return len(self._null_indices)

    def reset(
        self,
        objects: Dict[str, Object],
        action_skeleton: List,
        min_num_objects: Optional[int] = None,
        max_num_objects: Optional[int] = None,
    ) -> int:
        if min_num_objects is None:
            min_num_objects = 0
        if max_num_objects is None:
            max_num_objects = self.num_real_objects

        # Number of objects in this group that will be instantiated in the world.
        instances = [
            obj
            for obj in objects.values()
            if isinstance(obj, Variant) and obj.variants == self
        ]
        num_used = len(instances)

        # Remember objects that will be used in the action skeleton.
        self._required_instances = [
            obj
            for obj in instances
            if any(obj in primitive.arg_objects for primitive in action_skeleton)
        ]
        num_required = len(self._required_instances)

        # Compute valid range.
        hard_min = max(num_required, num_used - self.num_null_objects)
        hard_max = min(self.num_real_objects, num_used)
        min_num_objects = int(np.clip(min_num_objects, hard_min, hard_max))
        max_num_objects = int(np.clip(max_num_objects, hard_min, hard_max))
        max_num_objects = max(min_num_objects, max_num_objects)

        # Uniformly select the number of non-null objects to instantiate.
        num_objects = random.randint(min_num_objects, max_num_objects)
        num_null = num_used - num_objects

        # Reserve indices for required instances.
        self._reserved_indices = random.sample(self._real_indices, num_required)

        # Get indices for all other instances.
        self._available_indices = random.sample(
            set(self._real_indices) - set(self._reserved_indices),
            num_objects - num_required,
        ) + random.sample(self._null_indices, num_null)

        # Hide objects below table.
        for obj in self.objects:
            obj.disable_collisions()
            obj.set_pose(math.Pose(pos=np.array([0.0, 0.0, -0.5])))
            obj.freeze()

        return num_objects

    def pop_index(self, obj: Object, idx: Optional[int] = None) -> int:
        """Pops an index from the list of available indices.

        Args:
            obj: Object popping the index.
            idx: If specified, this index will be popped from the list.

        Returns:
            Popped index.
        """
        if obj in self._required_instances:
            index_pool = self._reserved_indices
        else:
            index_pool = self._available_indices

        if idx is not None:
            # Get the requested index.
            idx_pool = index_pool.index(idx)
        else:
            # Choose a random index from the ones remaining.
            idx_pool = random.randrange(len(index_pool))

        return index_pool.pop(idx_pool)

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[Object]:
        return iter(self.objects)

    def __getitem__(self, idx: Union[int, slice]):
        return self.objects[idx]

    def compute_probabilities(self, objects: Dict[str, Object]) -> None:
        # Number of non-null objects in this group.
        num_obj = sum(not isinstance(obj, Null) for obj in self.objects)

        # Total number of objects in this group.
        num_total = len(self.objects)

        # Number of objects in this group that will be instantiated by variants.
        num_used = sum(
            isinstance(obj, Variant) and obj.variants == self
            for obj in objects.values()
        )

        # Number of objects in this group that will remain uninstantiated.
        num_unused = num_total - num_used

        # Number of non-null objects in this group that will be instantiated.
        for num_obj_used in range(min(num_obj, num_used) + 1):
            # Number of non-null objects in this group that will remain uninstantiated.
            num_obj_unused = num_obj - num_obj_used

            num_comb_obj_used = math.comb(num_used, num_obj_used)
            num_comb_obj_unused = math.comb(num_unused, num_obj_unused)
            num_comb_obj = math.comb(num_total, num_obj)

            p_num_obj_used = num_comb_obj_used * num_comb_obj_unused / num_comb_obj

            print(f"p(num_obj_used = {num_obj_used}):", p_num_obj_used)


class Variant(WrapperObject):
    def __init__(
        self,
        physics_id: int,
        name: str,
        variants: Optional[List[Dict[str, Any]]] = None,
        group: Optional[str] = None,
        object_groups: Dict[str, ObjectGroup] = {},
    ):
        from temporal_policies.envs.pybullet.table.utils import load_config

        self.physics_id = physics_id
        self.name = name

        if variants is None and group is None:
            raise ValueError("One of variants or group must be specified")
        elif variants is not None and group is not None:
            raise ValueError("Only one of variants or group can be specified")

        if variants is not None:
            self._variants: Union[List[Object], ObjectGroup] = [
                Object.create(
                    physics_id=self.physics_id,
                    name=self.name,
                    **load_config(obj_config),
                )
                for obj_config in variants
            ]
            self._real_indices = [
                i
                for i, variant in enumerate(self.variants)
                if not variant.isinstance(Null)
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

    def set_variant(
        self, idx_variant: Optional[int], action_skeleton: List, lock: bool = False
    ) -> None:
        """Sets the variant for debugging purposes.

        Args:
            idx_variant: Index of the variant to set.
            action_skeleton: List of primitives used to identify required objects.
            lock: Whether to lock the variant so it remains the same upon resetting.
        """
        if isinstance(self.variants, ObjectGroup):
            idx_variant = self.variants.pop_index(self, idx_variant)
        else:
            if idx_variant is None:
                if any(self in primitive.arg_objects for primitive in action_skeleton):
                    idx_variant = random.choice(self._real_indices)
                else:
                    idx_variant = random.randrange(len(self.variants))

            # Hide unused variants below table.
            for i, obj in enumerate(self.variants):
                if i == idx_variant:
                    continue
                obj.disable_collisions()
                obj.set_pose(math.Pose(pos=np.array([0.0, 0.0, -0.5])))
                obj.freeze()

        self._body = self.variants[idx_variant]
        if lock:
            self._idx_variant = idx_variant

    def reset(self, action_skeleton: List) -> None:
        self.set_variant(self._idx_variant, action_skeleton)
        self.enable_collisions()
        self.unfreeze()
        self.body.reset(action_skeleton)
