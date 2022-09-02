import dataclasses
from typing import Dict, List, Optional, Sequence, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import symbolic
from shapely.geometry import Polygon

from temporal_policies.envs.pybullet.table.objects import Object, Null, Hook, Box, Rack
from temporal_policies.envs.pybullet.sim import math, body
from temporal_policies.envs.pybullet.sim.robot import Robot


dbprint = lambda *args: None  # noqa
# dbprint = print


AABB_EPS = 0.01
ALIGN_EPS = 0.99
TWIST_EPS = 0.001
TABLE_HEIGHT = 0.0
WORKSPACE_RADIUS = 0.7
WORKSPACE_MIN_X = 0.4
TIPPING_PROB = 0.1


def is_above(obj_a: Object, obj_b: Object) -> bool:
    """Returns True if the object a is above the object b."""
    min_child_z = obj_a.aabb()[0, 2]
    max_parent_z = obj_b.aabb()[1, 2]
    return min_child_z > max_parent_z - AABB_EPS


def is_upright(obj: Object) -> bool:
    """Returns True if the child objects z-axis aligns with the world frame."""
    aa = eigen.AngleAxisd(eigen.Quaterniond(obj.pose().quat))
    return abs(aa.axis.dot(np.array([0.0, 0.0, 1.0]))) >= ALIGN_EPS


def is_within_distance(
    obj_a: Object, obj_b: Object, distance: float, physics_id: int
) -> bool:
    """Returns True if the closest points between two objects are within distance."""
    return bool(
        p.getClosestPoints(
            obj_a.body_id, obj_b.body_id, distance, physicsClientId=physics_id
        )
    )


def is_moving(obj: Object) -> bool:
    """Returns True if the object is moving."""
    return bool((np.abs(obj.twist()) > TWIST_EPS).any())


def is_below_table(obj: Object) -> bool:
    """Returns True if the object is below the table."""
    return obj.pose().pos[2] < TABLE_HEIGHT


def is_touching(
    body_a: body.Body,
    body_b: body.Body,
    link_id_a: Optional[int] = None,
    link_id_b: Optional[int] = None,
) -> bool:
    """Returns True if there are any contact points between the two bodies."""
    assert body_a.physics_id == body_b.physics_id
    kwargs = {}
    if link_id_a is not None:
        kwargs["linkIndexA"] = link_id_a
    if link_id_b is not None:
        kwargs["linkIndexB"] = link_id_b
    contacts = p.getContactPoints(
        bodyA=body_a.body_id,
        bodyB=body_b.body_id,
        physicsClientId=body_a.physics_id,
        **kwargs,
    )
    return len(contacts) > 0


def compute_vertices(obj: Object, transform=False) -> np.ndarray:
    """Return object vertices and optionally transform to real world."""
    assert hasattr(obj, "size")
    dx, dy, dz = obj.size * 0.5    
    vertices = np.array(
        [[-dx, -dy, dz, 1],
        [-dx, dy, dz, 1],
        [dx, dy, dz, 1],
        [dx, -dy, dz, 1],]
    ).T
    if transform:
        vertices = obj.pose().to_eigen() * vertices
    return vertices[:3, :]


def is_intersecting(obj_a: Object, obj_b: Object) -> bool:
    """Returns True if object a intersects object b in the world x-y plane."""
    obj_a = compute_vertices(obj_a, transform=True)[:2].T
    obj_b = compute_vertices(obj_b, transform=True)[:2].T
    poly_a = Polygon(obj_a.tolist())
    poly_b = Polygon(obj_b.tolist())
    intersection = poly_a.intersects(poly_b)
    return intersection


def generate_grasp_pose(obj: Object, handlegrasp: bool = False) -> math.Pose:
    """Generates a grasp pose in the object frame of reference."""
    if obj.isinstance(Hook):
        hook: Hook = obj  # type: ignore
        pos_handle, pos_head, pos_joint = Hook.compute_link_positions(
            head_length=hook.head_length,
            handle_length=hook.handle_length,
            handle_y=hook.handle_y,
            radius=hook.radius,
        )
        if handlegrasp or np.random.random() < hook.handle_length / (
            hook.handle_length + hook.head_length
        ):
            # Handle.
            half_size = np.array([0.5 * hook.handle_length, hook.radius, hook.radius])
            if handlegrasp:
                xyz = pos_handle + np.random.uniform(-half_size, 0)
            else:
                xyz = pos_handle + np.random.uniform(-half_size, half_size)
            theta = 0.0
        else:
            # Head.
            half_size = np.array([hook.radius, 0.5 * hook.head_length, hook.radius])
            xyz = pos_head + np.random.uniform(-half_size, half_size)
            theta = np.pi / 2

        # Perturb angle by 10deg.
        theta += np.random.normal(scale=0.2)
        if theta > np.pi / 2:
            theta -= np.pi

        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    else:
        # Fit object between gripper fingers.
        max_aabb = 0.5 * obj.size
        max_aabb[:2] = np.minimum(max_aabb[:2], np.array([0.02, 0.02]))
        min_aabb = -0.5 * obj.size
        min_aabb = np.maximum(min_aabb, np.array([-0.02, -0.02, max_aabb[2] - 0.05]))

        xyz = np.random.uniform(min_aabb, max_aabb)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))

    return math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)


@dataclasses.dataclass
class Predicate:
    args: List[str]

    @classmethod
    def create(cls, proposition: str) -> "Predicate":
        predicate, args = symbolic.parse_proposition(proposition)
        predicate_classes = {
            name.lower(): predicate_class for name, predicate_class in globals().items()
        }
        predicate_class = predicate_classes[predicate]
        return predicate_class(args)

    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence["Predicate"]
    ) -> bool:
        """Generates a geometric grounding of a predicate."""
        dbprint(f"{self}.sample():", True)
        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence["Predicate"]
    ) -> bool:
        """Evaluates to True if the geometrically grounded predicate is satisfied."""
        dbprint(f"{self}.value():", True)
        return True

    def get_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        return [objects[arg] for arg in self.args]

    def __str__(self) -> str:
        return f"{type(self).__name__.lower()}({', '.join(self.args)})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class BeyondWorkspace(Predicate):
    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Evaluates to True if the object is beyond the robot workspace radius."""
        obj = self.get_arg_objects(objects)[0]
        distance = float(np.linalg.norm(obj.pose().pos[:2]))
        is_beyondworkspace = distance > WORKSPACE_RADIUS

        dbprint(f"{self}.value():", is_beyondworkspace, "distance:", distance)
        return is_beyondworkspace


class InWorkspace(Predicate):
    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Evaluates to True if the object is within the robot workspace."""
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        xyz = obj.pose().pos
        distance = float(np.linalg.norm(xyz[:2]))
        is_inworkspace = WORKSPACE_MIN_X < xyz[0] and distance <= WORKSPACE_RADIUS
        dbprint(f"{self}.value():", is_inworkspace, "x:", xyz[0], "distance:", distance)
        return is_inworkspace


class IsTippable(Predicate):
    """Unary predicate admitting non-upright configurations of an object."""

    pass


class Under(Predicate):
    """Unary predicate enforcing that an object be placed underneath another."""

    pass


class Free(Predicate):
    """Unary predicate enforcing that no top-down occlusions exist on the object."""
    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        child_obj = self.get_arg_objects(objects)[0]
        if child_obj.isinstance(Null):
            return True
        for obj in objects.values():
            if (not obj.isinstance(Null) 
                and not obj == child_obj
                and not is_above(child_obj, obj) 
                and is_intersecting(obj, child_obj)
            ):
                return False
        return True


class On(Predicate):
    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Samples a geometric grounding of the On(a, b) predicate."""
        child_obj, parent_obj = self.get_arg_objects(objects)

        if child_obj.is_static:
            dbprint(f"{self}.sample():", True, "- static child")
            return True
        if parent_obj.isinstance(Null):
            dbprint(f"{self}.sample():", False, "- null parent")
            return False

        xy_min = np.empty(2)
        xy_max = np.empty(2)
        z_max = parent_obj.aabb()[1, 2] + AABB_EPS
        T_parent_obj_to_world = parent_obj.pose()
        margin = 0.5 * child_obj.size[:2].max()

        if parent_obj.name == "table":
            is_under = False
            for obj in objects.values():
                if obj.isinstance(Rack) and f"under({child_obj}, {obj})" in state:
                    # Restrict placement location to under the rack
                    is_under = True
                    T_parent_obj_to_world = obj.pose()
                    xy_min[:2] = np.array([margin, margin]) - obj.size / 2
                    xy_max[:2] = obj.size[:2] - margin - obj.size / 2
                    break
            if not is_under:
                T_parent_obj_to_world = math.Pose()
                xyz_min, xyz_max = parent_obj.aabb()
                if f"beyondworkspace({child_obj})" in state:
                    # Increase the likelihood of sampling outside the workspace
                    r = WORKSPACE_RADIUS
                    xy_min[0] = r * np.cos(
                        np.arcsin(0.5 * (xyz_max[1] - xyz_min[1]) / r)
                    )
                    xy_max[0] = xyz_max[0] - margin
                    xy_min[1] = xyz_min[1] + margin
                    xy_max[1] = xyz_max[1] - margin
                elif f"inworkspace({child_obj})" in state:
                    # Increase the likelihood of sampling inside the workspace
                    xy_min[0] = WORKSPACE_MIN_X
                    xy_max[0] = WORKSPACE_RADIUS
                    xy_min[1] = xyz_min[1] + margin
                    xy_max[1] = xyz_max[1] - margin
                else:
                    xy_min[:2] = xyz_min[:2]
                    xy_max[:2] = xyz_max[:2]

        elif parent_obj.isinstance(Rack):
            xy_min[:2] = np.array([margin, margin]) - parent_obj.size[:2] / 2
            xy_max[:2] = parent_obj.size[:2] - margin - parent_obj.size[:2] / 2

        elif parent_obj.isinstance(Box):
            xy_min[:2] = np.array([margin, margin])
            xy_max[:2] = parent_obj.size[:2] - margin
            if np.any(xy_max - xy_min < 0):
                # Increase the likelihood of a stable placement location
                child_parent_ratio = child_obj.size[0] / parent_obj.size[0]
                x_min_ratio = min(0.25 * child_parent_ratio[0], 0.45)
                x_max_ratio = max(0.55, min(0.75 * child_parent_ratio[0], 0.95))
                y_min_ratio = min(0.25 * child_parent_ratio[1], 0.45)
                y_max_ratio = max(0.55, min(0.75 * child_parent_ratio[1], 0.95))
                xy_min[:2] = parent_obj.size[:2] * np.array([x_min_ratio, y_min_ratio])
                xy_max[:2] = parent_obj.size[:2] * np.array([x_max_ratio, y_max_ratio])
            xy_min -= parent_obj.size[:2] / 2
            xy_max -= parent_obj.size[:2] / 2
        else:
            raise ValueError(
                "[Predicate.On] parent object must be a table, rack, or box"
            )

        # Generate pose in parent coordinate frame
        xyz_parent_obj = np.zeros(3)
        xyz_parent_obj[:2] = np.random.uniform(xy_min, xy_max)

        # Convert pose to world coordinate frame (assumes parent in upright)
        xyz_world = T_parent_obj_to_world.to_eigen() * xyz_parent_obj

        # Ensure object is placed in specified region
        if f"beyondworkspace({child_obj})" in state:
            if xyz_world[0] < WORKSPACE_RADIUS:
                dbprint(f"{self}.sample():", False, "- should be beyond workspace")
                return False
        elif f"inworkspace({child_obj})" in state:
            if xyz_world[0] < WORKSPACE_MIN_X or xyz_world[0] > WORKSPACE_RADIUS:
                dbprint(f"{self}.sample():", False, "- should be in workspace")
                return False

        # Correct z-axis position
        xyz_world[2] = z_max + 0.5 * child_obj.size[2]
        if child_obj.isinstance(Rack):
            xyz_world[2] += 0.5 * child_obj.size[2]

        # Generate theta in the world coordinate frame
        theta = np.random.uniform(-np.pi, np.pi)
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        quat = eigen.Quaterniond(aa)

        if f"istippable({child_obj})" in state and not child_obj.isinstance(
            (Hook, Rack)
        ):
            # Tip the object over
            if np.random.random() < TIPPING_PROB:
                axis = np.random.uniform(-1, 1, size=2)
                axis /= np.linalg.norm(axis)
                quat = quat * eigen.Quaterniond(
                    eigen.AngleAxisd(np.pi / 2, np.array([*axis, 0.0]))
                )
                xyz_world[2] = z_max + 0.8 * child_obj.size[:2].max()

        pose = math.Pose(pos=xyz_world, quat=quat.coeffs)
        child_obj.set_pose(pose)

        dbprint(f"{self}.sample():", True)
        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Evaluates to True if the grounding of On(a, b) is geometrically valid."""
        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not is_above(child_obj, parent_obj):
            dbprint(f"{self}.value():", False, "- child below parent")
            return False

        # Ensure that object remains in specified region
        child_pos_x = child_obj.pose().pos[0]
        if f"beyondworkspace({child_obj})" in state:
            if child_pos_x < WORKSPACE_RADIUS:
                return False
        elif f"inworkspace({child_obj})" in state:
            if child_pos_x < WORKSPACE_MIN_X or child_pos_x > WORKSPACE_RADIUS:
                return False

        if f"istippable({child_obj})" not in state or child_obj.isinstance(
            (Hook, Rack)
        ):
            if not is_upright(child_obj):
                dbprint(f"{self}.value():", False, "- child not upright")
                return False

        dbprint(f"{self}.value():", True)
        return True


class HandleGrasp(Predicate):
    """Unary predicate enforcing a handle grasp on a hook object."""

    pass


class Inhand(Predicate):
    MAX_GRASP_ATTEMPTS = 1

    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Samples a geometric grounding of the InHand(a) predicate."""
        obj = self.get_arg_objects(objects)[0]
        if obj.is_static:
            dbprint(f"{self}.sample():", True, "- static")
            return True

        # Generate grasp pose.
        for i in range(Inhand.MAX_GRASP_ATTEMPTS):
            grasp_pose = generate_grasp_pose(obj, f"handlegrasp({obj})" in state)
            obj_pose = math.Pose.from_eigen(grasp_pose.to_eigen().inverse())
            obj_pose.pos += robot.home_pose.pos

            # Use fake grasp.
            obj.disable_collisions()
            obj.set_pose(obj_pose)
            robot.grasp_object(obj, realistic=False)
            obj.enable_collisions()

            # Make sure object isn't touching gripper.
            obj.unfreeze()
            p.stepSimulation(physicsClientId=robot.physics_id)
            if not is_touching(obj, robot):
                break
            elif i + 1 == Inhand.MAX_GRASP_ATTEMPTS:
                dbprint(f"{self}.sample():", False, "- exceeded max grasp attempts")
                return False

        dbprint(f"{self}.sample():", True)
        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """The geometric grounding of InHand(a) evaluates to True by construction."""
        return True
