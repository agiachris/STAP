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

    DISTANCE_MIN: Dict[Tuple[Type[Object], Type[Object]], float] = {
        (Box, Box): 0.05,
        (Box, Hook): 0.05,
        (Box, Rack): 0.1,
        (Hook, Rack): 0.1,
    }

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

            obj_a, obj_b = sorted(
                (child_obj.type(), obj.type()), key=lambda x: x.__name__
            )
            try:
                min_distance = Free.DISTANCE_MIN[(obj_a, obj_b)]
            except KeyError:
                continue
            if (
                (obj.isinstance(Rack) and f"beyondworkspace({obj})" in state)
                or f"infront({child_obj}, rack)" in state
                or f"infront({obj}, rack)" in state
            ):
                min_distance = 0.04
            if utils.is_within_distance(
                child_obj, obj, min_distance, obj.physics_id
            ) and not utils.is_above(child_obj, obj):
                dbprint(
                    f"{self}.value():",
                    False,
                    f"{child_obj} and {obj} are within min distance",
                )
                return False

        return True


class Tippable(Predicate):
    """Unary predicate admitting non-upright configurations of an object."""

    pass


class TableBounds:
    """Predicate that specifies minimum and maximum x-y bounds on the table."""

    MARGIN_SCALE: Dict[Type[Object], float] = {Hook: 0.25}

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds on the table as well as the modified margins."""
        assert parent_obj.name == "table"

        zone = type(self).__name__.lower()
        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            pos_bounds = poslimit.bounds(child_obj)
            zone = random.choice(list(pos_bounds.keys()))
            # Compute poslimit zone-specific angle
            if f"aligned({child_obj})" in state:
                theta = Aligned.sample_angle(obj=child_obj, zone=zone)
                child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
                margin = utils.compute_margins(child_obj)

            return pos_bounds[zone], margin

        elif f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj, zone=zone)
            child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
            margin = utils.compute_margins(child_obj)

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["table_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin

    @staticmethod
    def get_poslimit(
        obj: Object,
        state: Sequence[Predicate],
    ) -> Optional["PosLimit"]:
        try:
            idx_prop = state.index(f"poslimit({obj})")
        except ValueError:
            return None
        prop = state[idx_prop]
        assert isinstance(prop, PosLimit)
        return prop

    @classmethod
    def get_zone(
        cls,
        obj: Object,
        state: Sequence[Predicate],
    ) -> Optional["TableBounds"]:
        zones = [
            prop
            for prop in state
            if isinstance(prop, TableBounds) and prop.args[0] == obj
        ]
        if not zones and f"on({obj}, table)" in state:
            return cls()
        elif len(zones) == 1:
            return zones[0]
        elif len(zones) != 1:
            raise ValueError(f"{obj} cannot be in multiple zones: {zones}")

        return None

    @staticmethod
    def scale_margin(obj: Object, margins: np.ndarray) -> np.ndarray:
        try:
            bounds = TableBounds.MARGIN_SCALE[obj.type()]
        except KeyError:
            return margins
        return bounds * margins


class Aligned(Predicate):
    """Unary predicate enforcing that the object and world coordinate frames align."""

    ANGLE_EPS: float = 0.002
    ANGLE_STD: float = 0.05
    ANGLE_ABS: float = 0.1
    ZONE_ANGLES: Dict[Tuple[Type[Object], Optional[str]], float] = {
        (Rack, "inworkspace"): 0.5 * np.pi,
        (Rack, "beyondworkspace"): 0.0,
    }

    # def value(
    #     self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    # ) -> bool:
    #     obj = self.get_arg_objects(objects)[0]
    #     if obj.isinstance(Null):
    #         return True

    #     try:
    #         zone = TableBounds.get_zone(obj=obj, state=state)
    #         angle_mean = Aligned.ZONE_ANGLES[(obj.type(), type(zone).__name__.lower())]
    #         if (
    #             angle_mean - Aligned.ANGLE_ABS < -np.pi
    #             or angle_mean + Aligned.ANGLE_ABS > np.pi
    #         ):
    #             raise ValueError("Cannot recover wrapped angle.")
    #     except KeyError:
    #         angle_mean = 0.0

    #     angle = eigen.AngleAxisd(eigen.Quaterniond(obj.pose().quat)).angle - angle_mean
    #     if not (
    #         Aligned.ANGLE_EPS <= abs(angle) <= Aligned.ANGLE_ABS
    #         and utils.is_upright(obj)
    #     ):
    #         dbprint(f"{self}.value():", False)
    #         return False

    #     return True

    @staticmethod
    def sample_angle(obj: Object, zone: Optional[str] = None) -> float:
        angle = 0.0
        while abs(angle) < Aligned.ANGLE_EPS:
            angle = np.random.randn() * Aligned.ANGLE_STD

        try:
            angle_mu = Aligned.ZONE_ANGLES[(obj.type(), zone)]
        except KeyError:
            angle_mu = 0.0

        angle = np.clip(
            angle + angle_mu,
            angle_mu - Aligned.ANGLE_ABS,
            angle_mu + Aligned.ANGLE_ABS,
        )
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        return angle


class PosLimit(Predicate):
    """Unary predicate limiting the placement positions of particular object types."""

    POS_EPS: Dict[Type[Object], float] = {Rack: 0.01}
    POS_SPEC: Dict[Type[Object], Dict[str, np.ndarray]] = {
        Rack: {
            "inworkspace": np.array([0.44, -0.33]),
            "beyondworkspace": np.array([0.82, 0.00]),
        }
    }

    def bounds(self, child_obj: Object) -> Dict[str, np.ndarray]:
        assert child_obj.name == self.args[0]

        if child_obj.type() not in PosLimit.POS_SPEC:
            raise ValueError(f"Positions not specified for {child_obj.type()}")

        eps = PosLimit.POS_EPS[child_obj.type()]
        xys = PosLimit.POS_SPEC[child_obj.type()]
        bounds = {k: np.array([xy - eps, xy + eps]) for k, xy in xys.items()}
        return bounds


class InWorkspace(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in the robot workspace."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance((Null, Rack)):  # Rack is in workspace by construction.
            return True

        obj_pos = obj.pose().pos[:2]
        distance = float(np.linalg.norm(obj_pos))
        if not utils.is_inworkspace(obj_pos=obj_pos, distance=distance):
            dbprint(
                f"{self}.value():", False, "- pos:", obj_pos[:2], "distance:", distance
            )
            return False

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds inside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        zone = type(self).__name__.lower()
        if f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj, zone=zone)
            child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
            margin = utils.compute_margins(child_obj)

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            return poslimit.bounds(child_obj)[zone], margin

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["workspace_radius"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


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

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        margin = TableBounds.scale_margin(child_obj, margin)
        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["workspace_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["operational_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


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

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        margin = TableBounds.scale_margin(child_obj, margin)
        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["operational_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["operational_x_max"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


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

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        margin = TableBounds.scale_margin(child_obj, margin)
        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = utils.TABLE_CONSTRAINTS["obstruction_x_min"]
        xy_max[0] = utils.TABLE_CONSTRAINTS["workspace_radius"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class BeyondWorkspace(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in beyond the robot workspace."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        distance = float(np.linalg.norm(obj.pose().pos[:2]))
        if not utils.is_beyondworkspace(obj=obj, distance=distance):
            return False
            dbprint(f"{self}.value():", False, "- distance:", distance)

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds outside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        zone = type(self).__name__.lower()
        if f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj, zone=zone)
            child_obj.set_pose(utils.compute_object_pose(child_obj, theta))
            margin = utils.compute_margins(child_obj)

        poslimit = TableBounds.get_poslimit(child_obj, state)
        if poslimit is not None:
            return poslimit.bounds(child_obj)[zone], margin

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        r = utils.TABLE_CONSTRAINTS["workspace_radius"]
        xy_min[0] = r * np.cos(np.arcsin(0.5 * (xy_max[1] - xy_min[1]) / r))
        xy_min += margin
        xy_max -= margin

        return bounds, margin


class InOodZone(Predicate, TableBounds):
    """Unary predicate ensuring than an object is in beyond the robot workspace."""

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self.get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        return True

    def get_bounds_and_margin(
        self,
        child_obj: Object,
        parent_obj: Object,
        state: Sequence[Predicate],
        margin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum x-y bounds outside the workspace."""
        assert child_obj.name == self.args[0] and parent_obj.name == "table"

        bounds = parent_obj.aabb()[:, :2]
        xy_min, xy_max = bounds
        xy_min[0] = bounds[0, 0]
        xy_max[0] = utils.TABLE_CONSTRAINTS["table_x_min"]
        xy_min += margin
        xy_max -= margin

        return bounds, margin


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
    def bounds(
        child_obj: Object,
        parent_obj: Object,
        margin: np.ndarray = np.zeros(2),
    ) -> np.ndarray:
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

    PULL_MARGIN: Dict[Tuple[Type[Object], Type[Object]], Dict[Optional[str], float]] = {
        (Box, Rack): {"inworkspace": 3.0, "beyondworkspace": 1.5},
        (Box, Box): {"inworkspace": 3.0, "beyondworkspace": 3.0},
        (Rack, Hook): {"inworkspace": 0.25, "beyondworkspace": 0.25},
    }

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        target_obj, intersect_obj = self.get_arg_objects(objects)
        if target_obj.isinstance(Null) or intersect_obj.isinstance(Null):
            return True

        target_line = LineString([[0, 0], target_obj.pose().pos[:2].tolist()])
        if intersect_obj.isinstance(Hook):
            convex_hulls = Object.convex_hulls(intersect_obj, project_2d=True)
        else:
            convex_hulls = intersect_obj.convex_hulls(world_frame=True, project_2d=True)

        if len(convex_hulls) > 1:
            raise NotImplementedError(f"Compound shapes are not yet supported")
        vertices = convex_hulls[0]

        try:
            pull_margins = NonBlocking.PULL_MARGIN[
                (target_obj.type(), intersect_obj.type())
            ]
        except KeyError:
            pull_margins = None

        if pull_margins is not None:
            if utils.is_inworkspace(obj=intersect_obj):
                zone = "inworkspace"
            elif utils.is_beyondworkspace(obj=intersect_obj):
                zone = "beyondworkspace"
            else:
                zone = None
            try:
                margin_scale = pull_margins[zone]
            except KeyError:
                margin_scale = 1
            target_margin = margin_scale * target_obj.size[:2].max()
            # Expand the vertices by the margin.
            center = vertices.mean(axis=0)
            vertices += np.sign(vertices - center) * target_margin

        intersect_poly = Polygon(vertices)
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
        if f"aligned({child_obj})" in state:
            theta = Aligned.sample_angle(obj=child_obj)
        else:
            theta = np.random.uniform(-np.pi, np.pi)
        child_obj.set_pose(utils.compute_object_pose(child_obj, theta))

        # Determine object margins after rotating
        margin_world_frame = utils.compute_margins(child_obj)

        try:
            rack_obj = next(obj for obj in objects.values() if obj.isinstance(Rack))
        except StopIteration:
            rack_obj = None

        if (
            parent_obj.name == "table"
            and rack_obj is not None
            and f"under({child_obj}, {rack_obj})" in state
        ):
            # Restrict placement location to under the rack
            parent_obj = rack_obj

        # Determine stable sampling regions on parent surface
        if parent_obj.name == "table":
            zone = TableBounds.get_zone(obj=child_obj, state=state)
            if zone is not None:
                bounds, margin_world_frame = zone.get_bounds_and_margin(
                    child_obj=child_obj,
                    parent_obj=parent_obj,
                    state=state,
                    margin=margin_world_frame,
                )
                xy_min, xy_max = bounds

            if rack_obj is not None and f"infront({child_obj}, {rack_obj})" in state:
                infront_bounds = InFront.bounds(
                    child_obj=child_obj, parent_obj=rack_obj, margin=margin_world_frame
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
            if isinstance(prop, (Free, TableBounds)) and prop.args[-1] == child_obj.name
        ]

        samples = 0
        success = False
        quat_np = child_obj.pose().quat
        T_parent_obj_to_world = parent_obj.pose().to_eigen()
        while not success and samples < len(range(On.MAX_SAMPLE_ATTEMPTS)):
            # Generate pose and convert to world frame (assumes parent in upright)
            quat = eigen.Quaterniond(quat_np)
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
    "inoodzone": InOodZone,
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
    "inoodzone",
    "under",
    "infront",
    "nonblocking",
    "on",
    "inhand",
]


assert len(UNARY_PREDICATES) + len(BINARY_PREDICATES) == len(PREDICATE_HIERARCHY)
