import dataclasses
from typing import Optional, Dict, List, Sequence, Tuple, Type

import random
from ctrlutils import eigen
import numpy as np
import pybullet as p
import symbolic
from shapely.geometry import Polygon, LineString

from temporal_policies.envs.pybullet.table import primitive_actions, utils
from temporal_policies.envs.pybullet.table.objects import Box, Hook, Null, Object, Rack
from temporal_policies.envs.pybullet.sim import math
from temporal_policies.envs.pybullet.sim.robot import Robot


dbprint = lambda *args: None  # noqa
# dbprint = print


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
        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence["Predicate"]
    ) -> bool:
        """Evaluates to True if the geometrically grounded predicate is satisfied."""
        return True

    def get_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        return [objects[arg] for arg in self.args]

    def __str__(self) -> str:
        return f"{type(self).__name__.lower()}({', '.join(self.args)})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class HandleGrasp(Predicate):
    """Unary predicate enforcing a handle grasp towards the tail end on a hook object."""

    pass


class UpperHandleGrasp(Predicate):
    """Unary predicate enforcing a handle grasp towards the head on a hook object."""

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
            if f"inhand({obj})" in state or obj.isinstance(Null) or obj == child_obj:
                continue
            if utils.is_under(child_obj, obj):
                dbprint(f"{self}.value():", False, f"{child_obj} under {obj}")
                return False
        return True


class Aligned(Predicate):
    """Unary predicate enforcing that the object and world coordinate frames align."""

    ANGLE_EPS = 0.002
    ANGLE_STD = 0.05
    ANGLE_ABS = 0.1

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        angle = eigen.AngleAxisd(eigen.Quaterniond(obj.pose().quat)).angle
        if not (
            Aligned.ANGLE_EPS <= abs(angle) <= Aligned.ANGLE_ABS
            and utils.is_upright(obj)
        ):
            dbprint(f"{self}.value():", False)
            return False

        return True

    @staticmethod
    def sample_angle() -> float:
        angle = 0.0
        while abs(angle) < Aligned.ANGLE_EPS:
            angle = np.random.randn() * Aligned.ANGLE_STD
        return np.clip(angle, -Aligned.ANGLE_ABS, Aligned.ANGLE_ABS)


class Tippable(Predicate):
    """Unary predicate admitting non-upright configurations of an object."""

    pass


class TableBounds:
    """Predicate that specifies minimum and maximum x-y bounds on the table."""

    def bounds(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        """Returns the minimum and maximum x-y bounds on the table."""
        assert parent_obj.name == "table"

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            pos_bounds = poslimit.bounds(child_obj)
            return random.choice(pos_bounds)

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["table_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds

    @staticmethod
    def get_poslimit(
        child_obj: Object,
        state: Sequence[Predicate],
    ) -> Optional["PosLimit"]:
        try:
            prop = state[state.index(f"poslimit({child_obj})")]
            assert isinstance(prop, PosLimit)
            return prop
        except ValueError:
            return None


class PosLimit(Predicate):
    """Unary predicate limiting the placement positions of particular object types."""

    POS_EPS: Dict[Type[Object], float] = {Rack: 0.01}
    POS_SPEC: Dict[Type[Object], List[np.ndarray]] = {
        Rack: [np.array([0.50, -0.25]), np.array([0.82, 0.00])],
    }

    def bounds(self, child_obj: Object) -> List[np.ndarray]:
        assert child_obj.name == self.args[0]

        if child_obj.type() not in PosLimit.POS_SPEC:
            raise ValueError(f"Positions not specified for {child_obj.type()}")

        eps = PosLimit.POS_EPS[child_obj.type()]
        xys = PosLimit.POS_SPEC[child_obj.type()]
        bounds = [np.array([xy - eps, xy + eps]) for xy in xys]

        return bounds


class InWorkspace(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in the robot workspace."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose().pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not utils.is_inworkspace(obj_pos=obj_pos, distance=distance):
            dbprint(f"{self}.value():", False, "- pos:", obj.pose().pos[:2], "distance:", distance)
            return False            

        return True

    def bounds(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        """Returns the minimum and maximum x-y bounds inside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            return poslimit.bounds(child_obj)[0]

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["workspace_radius"]
        xy_min += margin
        xy_max -= margin

        return bounds


class InCollisionZone(Predicate, TableBounds):
    """Unary predicate ensuring the object is in the collision zone."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose().pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not (
            utils.TABLE_CONSTRAINTS["workspace_x_min"]
            <= obj.pose().pos[0]
            < utils.TABLE_CONSTRAINTS["operational_x_min"]
            and distance < utils.TABLE_CONSTRAINTS["workspace_radius"]
        ):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos, "distance:", distance)
            return False

        return True

    def bounds(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["operational_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds


class InOperationalZone(Predicate, TableBounds):
    """Unary predicate ensuring the object is in the operational zone."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose().pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not (
            utils.TABLE_CONSTRAINTS["operational_x_min"]
            <= obj_pos[0]
            < utils.TABLE_CONSTRAINTS["operational_x_max"]
            and distance < utils.TABLE_CONSTRAINTS["workspace_radius"]
        ):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos, "distance:", distance)
            return False

        return True

    def bounds(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["operational_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["operational_x_max"]
        xy_min += margin
        xy_max -= margin

        return bounds


class InObstructionZone(Predicate, TableBounds):
    """Unary predicate ensuring the object is in the obstruction zone."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        obj_pos = obj.pose().pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not (
            obj_pos[0] >= utils.TABLE_CONSTRAINTS["obstruction_x_min"]
            and distance < utils.TABLE_CONSTRAINTS["workspace_radius"]
        ):
            dbprint(f"{self}.value():", False, "- pos:", obj_pos, "distance:", distance)
            return False

        return True

    def bounds(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["obstruction_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["workspace_radius"]
        xy_min += margin
        xy_max -= margin

        return bounds


class BeyondWorkspace(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in beyond the robot workspace."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        distance = float(np.linalg.norm(obj.pose().pos[:2]))
        if distance < utils.TABLE_CONSTRAINTS["workspace_radius"]:
            dbprint(f"{self}.value():", False, "- distance:", distance)
            return False

        return True

    def bounds(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
        """Returns the minimum and maximum x-y bounds outside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            return poslimit.bounds(child_obj)[1]

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        r = utils.TABLE_CONSTRAINTS["workspace_radius"]
        xy_min[0] = r * np.cos(np.arcsin(0.5 * (xy_max[1] - xy_min[1]) / r))
        xy_min += margin
        xy_max -= margin

        return bounds


class Inhand(Predicate):
    MAX_GRASP_ATTEMPTS = 1

    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Samples a geometric grounding of the InHand(a) predicate."""
        obj = self.get_arg_objects(objects)[0]
        if obj.is_static:
            return True

        # Generate grasp pose.
        for i in range(Inhand.MAX_GRASP_ATTEMPTS):
            grasp_pose = self.generate_grasp_pose(
                obj,
                handlegrasp=f"handlegrasp({obj})" in state,
                upperhandlegrasp=f"upperhandlegrasp({obj})" in state,
            )
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
            if not utils.is_touching(obj, robot):
                break
            elif i + 1 == Inhand.MAX_GRASP_ATTEMPTS:
                dbprint(f"{self}.sample():", False, "- exceeded max grasp attempts")
                return False

        dbprint(f"{self}.sample():", True)
        return True

    @staticmethod
    def generate_grasp_pose(
        obj: Object, handlegrasp: bool = False, upperhandlegrasp: bool = False
    ) -> math.Pose:
        """Generates a grasp pose in the object frame of reference."""
        # Maximum deviation of the object from the gripper's center y.
        MAX_GRASP_Y_OFFSET = 0.01
        # Gap required between control point and object bottom.
        FINGER_COLLISION_MARGIN = 0.02
        FINGER_WIDTH = 0.022
        FINGER_HEIGHT = 0.04
        FINGER_DISTANCE = 0.08
        THETA_STDDEV = 0.05

        if obj.isinstance(Hook):
            hook: Hook = obj  # type: ignore
            pos_handle, pos_head, pos_joint = Hook.compute_link_positions(
                head_length=hook.head_length,
                handle_length=hook.handle_length,
                handle_y=hook.handle_y,
                radius=hook.radius,
            )
            if (
                handlegrasp
                or upperhandlegrasp
                or np.random.random()
                < hook.handle_length / (hook.handle_length + hook.head_length)
            ):
                # Handle.
                min_xyz, max_xyz = np.array(obj.bbox)

                if upperhandlegrasp:
                    min_xyz[0] = 0.0
                min_xyz[1] = pos_handle[1] - MAX_GRASP_Y_OFFSET
                min_xyz[2] += FINGER_COLLISION_MARGIN

                max_xyz[0] = pos_head[0] - hook.radius - 0.5 * FINGER_WIDTH
                if handlegrasp:
                    max_xyz[0] = 0.0
                max_xyz[1] = pos_handle[1] + MAX_GRASP_Y_OFFSET

                theta = 0.0
            else:
                # Head.
                min_xyz, max_xyz = np.array(obj.bbox)

                min_xyz[0] = pos_head[0] - MAX_GRASP_Y_OFFSET
                if hook.handle_y < 0:
                    min_xyz[1] = pos_handle[1] + hook.radius + 0.5 * FINGER_WIDTH
                min_xyz[2] += FINGER_COLLISION_MARGIN

                max_xyz[0] = pos_head[0] + MAX_GRASP_Y_OFFSET
                if hook.handle_y > 0:
                    max_xyz[1] = pos_handle[1] - hook.radius - 0.5 * FINGER_WIDTH

                theta = np.pi / 2
        else:
            # Fit object between gripper fingers.
            theta = np.random.choice([0.0, np.pi / 2])

            min_xyz, max_xyz = np.array(obj.bbox)
            if theta == 0.0:
                y_center = 0.5 * (min_xyz[1] + max_xyz[1])
                min_xyz[1] = max(
                    min_xyz[1] + 0.5 * FINGER_DISTANCE, y_center - MAX_GRASP_Y_OFFSET
                )
                max_xyz[1] = min(
                    max_xyz[1] - 0.5 * FINGER_DISTANCE, y_center + MAX_GRASP_Y_OFFSET
                )
            elif theta == np.pi / 2:
                x_center = 0.5 * (min_xyz[0] + max_xyz[0])
                min_xyz[0] = max(
                    min_xyz[0] + 0.5 * FINGER_DISTANCE, x_center - MAX_GRASP_Y_OFFSET
                )
                max_xyz[0] = min(
                    max_xyz[0] - 0.5 * FINGER_DISTANCE, x_center + MAX_GRASP_Y_OFFSET
                )

            min_xyz[2] += FINGER_COLLISION_MARGIN
            min_xyz[2] = max(min_xyz[2], max_xyz[0] - FINGER_HEIGHT)

        xyz = np.random.uniform(min_xyz, max_xyz)
        theta += np.random.normal(scale=THETA_STDDEV)
        theta = np.clip(theta, *primitive_actions.PickAction.RANGES["theta"])
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))

        return math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)


class Under(Predicate):
    """Unary predicate enforcing that an object be placed underneath another."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not utils.is_under(child_obj, parent_obj):
            dbprint(f"{self}.value():", False)
            return False

        return True


class InFront(Predicate):
    """Binary predicate enforcing that one object is in-front of another with
    respect to the world x-y coordinate axis."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        child_pos = child_obj.pose().pos
        xy_min, xy_max = parent_obj.aabb()[:, :2]
        if (
            child_pos[0] >= xy_min[0]
            or child_pos[1] <= xy_min[1]
            or child_pos[1] >= xy_max[1]
            or utils.is_under(child_obj, parent_obj)
        ):
            dbprint(f"{self}.value():", False, "- pos:", child_pos)
            return False

        return True

    @staticmethod
    def bounds(parent_obj: Object, margin: np.ndarray = np.zeros(2)) -> np.ndarray:
        """Returns the minimum and maximum x-y bounds in front of the parent object."""
        assert parent_obj.isinstance(Rack)

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_max[0] = xy_min[0]
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds


class NonBlocking(Predicate):
    """Binary predicate ensuring that one object is not occupying a straightline
    path from the robot base to another object."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        target_obj, intersect_obj = self.get_arg_objects(objects)
        if target_obj.isinstance(Null) or intersect_obj.isinstance(Null):
            return True

        target_line = LineString([[0, 0], target_obj.pose().pos[:2].tolist()])
        vertices = np.concatenate(
            intersect_obj.convex_hulls(world_frame=True, project_2d=True), axis=0
        )
        intersect_poly = Polygon(vertices.tolist())
        if intersect_poly.intersects(target_line):
            dbprint(f"{self}.value():", False)
            return False

        return True


class On(Predicate):
    MAX_SAMPLE_ATTEMPTS = 10

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

        # Parent surface height
        parent_z = parent_obj.aabb()[1, 2] + utils.EPSILONS["aabb"]

        # Generate theta in the world coordinate frame
        theta = (
            Aligned.sample_angle()
            if f"aligned({child_obj})" in state
            else np.random.uniform(-np.pi, np.pi)
        )
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        quat = eigen.Quaterniond(aa)

        # Determine object margins after rotating
        pre_pose = math.Pose(pos=child_obj.pose().pos, quat=quat.coeffs)
        child_obj.set_pose(pre_pose)
        child_aabb = child_obj.aabb()[:, :2]
        margin_world_frame = 0.5 * np.array(
            [child_aabb[1, 0] - child_aabb[0, 0], child_aabb[1, 1] - child_aabb[0, 1]]
        )

        # Determine stable sampling regions on parent surface
        if parent_obj.name == "table":
            try:
                rack_obj = next(obj for obj in objects.values() if obj.isinstance(Rack))
            except StopIteration:
                rack_obj = None

            if rack_obj is not None and f"under({child_obj}, {rack_obj})" in state:
                # Restrict placement location to under the rack
                parent_obj = rack_obj

            zones = [
                prop
                for prop in state
                if isinstance(prop, TableBounds) and prop.args[0] == child_obj
            ]
            if len(zones) == 0:
                zone = TableBounds()
            elif len(zones) != 1:
                raise ValueError(f"{child_obj} cannot be in multiple zones: {zones}")
            else:
                zone = zones[0]

            if child_obj.isinstance(Hook) and isinstance(
                zone, (InCollisionZone, InOperationalZone, InObstructionZone)
            ):
                # Scale down margins in tight spaces
                margin_world_frame *= 0.25

            bounds = zone.bounds(
                child_obj=child_obj,
                parent_obj=parent_obj,
                state=state,
                margin=margin_world_frame,
            )
            xy_min, xy_max = bounds

            if rack_obj is not None and f"infront({child_obj}, {rack_obj})" in state:
                infront_bounds = InFront.bounds(
                    parent_obj=rack_obj, margin=margin_world_frame
                )
                intersection = self.compute_bound_intersection(bounds, infront_bounds)
                if intersection is None:
                    dbprint(
                        f"{self}.sample():",
                        False,
                        f"- no intersection between infront({child_obj}, {rack_obj}) and {zone}",
                    )
                    return False
                xy_min, xy_max = intersection

        elif parent_obj.isinstance((Rack, Box)):
            xy_min, xy_max = self.compute_stable_region(child_obj, parent_obj)

        else:
            raise ValueError(
                "[Predicate.On] parent object must be a table, rack, or box"
            )

        # Obtain predicates to validate sampled pose
        propositions = [
            prop
            for prop in state
            if isinstance(prop, (Free, TableBounds, NonBlocking))
            and prop.args[-1] == child_obj.name
        ]

        samples = 0
        success = False
        T_parent_obj_to_world = parent_obj.pose().to_eigen()
        while not success and samples < len(range(On.MAX_SAMPLE_ATTEMPTS)):
            # Generate pose and convert to world frame (assumes parent in upright)
            xyz_parent_frame = np.zeros(3)
            xyz_parent_frame[:2] = np.random.uniform(xy_min, xy_max)
            xyz_world_frame = T_parent_obj_to_world * xyz_parent_frame
            xyz_world_frame[2] = parent_z + 0.5 * child_obj.size[2]
            if child_obj.isinstance(Rack):
                xyz_world_frame[2] += 0.5 * child_obj.size[2]

            if f"tippable({child_obj})" in state and not child_obj.isinstance(
                (Hook, Rack)
            ):
                # Tip the object over
                if np.random.random() < utils.EPSILONS["tipping"]:
                    axis = np.random.uniform(-1, 1, size=2)
                    axis /= np.linalg.norm(axis)
                    quat = quat * eigen.Quaterniond(
                        eigen.AngleAxisd(np.pi / 2, np.array([*axis, 0.0]))
                    )
                    xyz_world_frame[2] = parent_z + 0.8 * child_obj.size[:2].max()

            pose = math.Pose(pos=xyz_world_frame, quat=quat.coeffs)
            child_obj.set_pose(pose)

            if any(not prop.value(robot, objects, state) for prop in propositions):
                samples += 1
                continue
            success = True

        dbprint(f"{self}.sample():", success)
        return success

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        """Evaluates to True if the grounding of On(a, b) is geometrically valid."""
        child_obj, parent_obj = self.get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not utils.is_above(child_obj, parent_obj):
            dbprint(f"{self}.value():", False, "- child below parent")
            return False

        if f"tippable({child_obj})" not in state and not utils.is_upright(child_obj):
            dbprint(f"{self}.value():", False, "- child not upright")
            return False

        return True

    @staticmethod
    def compute_stable_region(
        child_obj: Object,
        parent_obj: Object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heuristically compute stable placement region on parent object."""
        # Compute child aabb in parent object frame
        R_child_to_world = child_obj.pose().to_eigen().matrix[:3, :3]
        R_world_to_parent = parent_obj.pose().to_eigen().inverse().matrix[:3, :3]
        vertices = np.concatenate(child_obj.convex_hulls(), axis=0).T
        vertices = R_world_to_parent @ R_child_to_world @ vertices
        child_aabb = np.array([vertices.min(axis=1), vertices.max(axis=1)])

        # Compute margin in the parent frame
        margin = 0.5 * np.array(
            [child_aabb[1, 0] - child_aabb[0, 0], child_aabb[1, 1] - child_aabb[0, 1]]
        )
        xy_min = margin
        xy_max = parent_obj.size[:2] - margin
        if np.any(xy_max - xy_min <= 0):
            # Increase the likelihood of a stable placement location
            child_parent_ratio = 2 * margin / parent_obj.size[:2]
            x_min_ratio = min(0.25 * child_parent_ratio[0], 0.45)
            x_max_ratio = max(0.55, min(0.75 * child_parent_ratio[0], 0.95))
            y_min_ratio = min(0.25 * child_parent_ratio[1], 0.45)
            y_max_ratio = max(0.55, min(0.75 * child_parent_ratio[1], 0.95))
            xy_min[:2] = parent_obj.size[:2] * np.array([x_min_ratio, y_min_ratio])
            xy_max[:2] = parent_obj.size[:2] * np.array([x_max_ratio, y_max_ratio])

        xy_min -= 0.5 * parent_obj.size[:2]
        xy_max -= 0.5 * parent_obj.size[:2]
        return xy_min, xy_max

    @staticmethod
    def compute_bound_intersection(*bounds: np.ndarray) -> Optional[np.ndarray]:
        """Compute intersection of a sequence of xy_min and xy_max bounds."""
        stacked_bounds = np.array(bounds)
        xy_min = stacked_bounds[:, 0].max(axis=0)
        xy_max = stacked_bounds[:, 1].min(axis=0)

        if not (xy_max - xy_min > 0).all():
            return None

        return np.array([xy_min, xy_max])


UNARY_PREDICATES = {
    "handlegrasp": HandleGrasp,
    "upperhandlegrasp": UpperHandleGrasp,
    "free": Free,
    "aligned": Aligned,
    "tippable": Tippable,
    "poslimit": PosLimit,
    "inworkspace": InWorkspace,
    "incollisionzone": InCollisionZone,
    "inoperationalzone": InOperationalZone,
    "inobstructionzone": InObstructionZone,
    "beyondworkspace": BeyondWorkspace,
    "inhand": Inhand,
}


BINARY_PREDICATES = {
    "under": Under,
    "infront": InFront,
    "nonblocking": NonBlocking,
    "on": On,
}


PREDICATE_HIERARCHY = [
    "handlegrasp",
    "upperhandlegrasp",
    "free",
    "aligned",
    "tippable",
    "poslimit",
    "inworkspace",
    "incollisionzone",
    "inoperationalzone",
    "inobstructionzone",
    "beyondworkspace",
    "under",
    "infront",
    "nonblocking",
    "on",
    "inhand",
]


assert len(UNARY_PREDICATES) + len(BINARY_PREDICATES) == len(PREDICATE_HIERARCHY)
