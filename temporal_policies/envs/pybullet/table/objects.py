import dataclasses
import random
from typing import Any, Dict, List, Optional, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import spatialdyn as dyn
import symbolic

from temporal_policies.envs.pybullet.sim import body, math, shapes
from temporal_policies.envs.pybullet.sim.robot import ControlException, Robot
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

    def reset(self, robot: Robot, objects: Dict[str, "Object"]) -> None:
        if self.is_static or self.initial_state is None:
            return

        predicate, args = symbolic.parse_proposition(self.initial_state)
        if predicate == "on":
            parent_obj = objects[args[1]]

            # Generate pose on parent.
            xyz_min, xyz_max = parent_obj.aabb()
            xyz = np.zeros(3)
            xyz[:2] = np.random.uniform(0.9 * xyz_min[:2], 0.9 * xyz_max[:2])
            xyz[2] = xyz_max[2] + 0.2
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
            pose = math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)

            self.set_pose(pose)

        elif predicate == "inhand":
            obj = objects[args[0]]
            self.disable_collisions()

            # Retry grasps.
            for _ in range(5):
                # Generate grasp pose.
                xyz = np.array(robot.home_pose.pos)
                xyz += 0.45 * np.random.uniform(-obj.size, obj.size)
                theta = np.random.uniform(-np.pi / 2, np.pi / 2)
                aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
                pose = math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)

                # Generate post-pick pose.
                table_xyz_min, table_xyz_max = objects["table"].aabb()
                xyz_pick = np.array([0.0, 0.0, obj.size[2] + 0.1])
                xyz_pick[:2] = np.random.uniform(
                    0.9 * table_xyz_min[:2], 0.9 * table_xyz_max[:2]
                )

                # Use fake grasp.
                self.set_pose(pose)
                robot.grasp_object(obj, realistic=False)
                try:
                    robot.goto_pose(pos=xyz_pick)
                except ControlException:
                    robot.reset()
                    continue

                break

            self.enable_collisions()
        else:
            raise NotImplementedError

    @classmethod
    def create(
        cls, physics_id: int, object_type: str, object_kwargs: Dict[str, Any], **kwargs
    ) -> "Object":
        object_class = globals()[object_type]
        return object_class(physics_id, **object_kwargs, **kwargs)

    def isinstance(self, class_or_tuple: type) -> bool:
        return isinstance(self, class_or_tuple)


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
                pos=np.array([-radius / 2, handle_y * head_length / 2 - dy, 0.0]),
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
                pos=np.array([(handle_length - radius) / 2, -dy, 0.0]),
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
                pos=np.array(
                    [(handle_length - radius) / 2, handle_y * head_length / 2 - dy, 0.0]
                )
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

        self._size = np.array(
            [handle_length + radius, head_length + 2 * abs(dy), 2 * radius]
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
    def size(self) -> np.ndarray:
        return self._size

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

    def aabb(self) -> np.ndarray:
        raise NotImplementedError


class Union(Object):
    def __init__(
        self,
        physics_id: int,
        name: str,
        options: List[Dict[str, Any]],
        initial_state: Optional[str] = None,
    ):
        self.physics_id = physics_id
        self.name = name
        self.initial_state = initial_state

        self._objects = [
            Object.create(
                physics_id=self.physics_id,
                name=self.name,
                initial_state=self.initial_state,
                **obj_kwargs,
            )
            for obj_kwargs in options
        ]
        self._masses = [
            p.getDynamicsInfo(obj.body_id, -1, physicsClientId=self.physics_id)[0]
            for obj in self._objects
        ]
        self._body = None

    @property
    def body(self) -> Object:
        if self._body is None:
            raise RuntimeError("Union.reset() must be called first")
        return self._body

    def reset(self, robot: Robot, objects: Dict[str, "Object"]) -> None:
        idx_body = random.randrange(len(self._objects))
        for i, body in enumerate(self._objects):
            if i == idx_body:
                continue
            body.disable_collisions()
            body.set_pose(math.Pose(pos=np.array([0.0, 0.0, -0.5])))
            body.freeze()

        self._body = self._objects[idx_body]
        self.body.enable_collisions()
        self.body.unfreeze()
        self.body.reset(robot, objects)

    def isinstance(self, class_or_tuple: type) -> bool:
        return self.body.isinstance(class_or_tuple)

    # Body methods.

    @property
    def body_id(self) -> int:
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
    def is_static(self) -> bool:
        return self.body.is_static

    # Box methods.

    @property
    def size(self) -> np.ndarray:
        return self.body.size

    # Hook methods.

    @property
    def head_length(self) -> float:
        return self.body.head_length

    @property
    def handle_length(self) -> float:
        return self.body.handle_length

    @property
    def handle_y(self) -> float:
        return self.body.handle_y

    @property
    def bbox(self) -> np.ndarray:
        return self.body.bbox
