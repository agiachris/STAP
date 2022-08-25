import dataclasses
from typing import Dict, List, Sequence

from ctrlutils import eigen
import numpy as np
import pybullet as p
import symbolic

from temporal_policies.envs.pybullet.table.objects import Hook, Null, Object
from temporal_policies.envs.pybullet.sim import math
from temporal_policies.envs.pybullet.sim.robot import ControlException, Robot


def is_above(child_aabb: np.ndarray, parent_aabb: np.ndarray) -> bool:
    return child_aabb[0, 2] > parent_aabb[1, 2] - 0.01


def is_upright(quat: np.ndarray) -> bool:
    aa = eigen.AngleAxisd(eigen.Quaterniond(quat))
    return abs(aa.axis.dot(np.array([0.0, 0.0, 1.0]))) >= 0.99


def is_within_distance(
    body_id_a: int, body_id_b: int, distance: float, physics_id: int
) -> bool:
    return bool(
        p.getClosestPoints(body_id_a, body_id_b, distance, physicsClientId=physics_id)
    )


def is_moving(twist: np.ndarray) -> bool:
    return bool((np.abs(twist) > 0.001).any())


def is_below_table(pos: np.ndarray) -> bool:
    return pos[2] < 0.0


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
        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence["Predicate"]
    ) -> bool:
        return True

    def _get_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        return [objects[arg] for arg in self.args]

    def __str__(self) -> str:
        return f"{type(self).__name__.lower()}({', '.join(self.args)})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class BeyondWorkspace(Predicate):
    WORKSPACE_RADIUS = 0.75

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self._get_arg_objects(objects)[0]
        distance = float(np.linalg.norm(obj.pose().pos[:2]))
        return distance > self.WORKSPACE_RADIUS


class InWorkspace(Predicate):
    MIN_X = 0.4

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self._get_arg_objects(objects)[0]
        if obj.isinstance(Null):
            return True

        xyz = obj.pose().pos
        distance = float(np.linalg.norm(xyz[:2]))
        return self.MIN_X < xyz[0] and distance <= BeyondWorkspace.WORKSPACE_RADIUS


class IsTippable(Predicate):
    pass


class On(Predicate):
    TIPPED_PROB = 0.1

    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        child_obj, parent_obj = self._get_arg_objects(objects)
        if child_obj.is_static:
            return True

        # Get parent aabb.
        xyz_min, xyz_max = parent_obj.aabb()
        if parent_obj.name == "table":
            if f"beyondworkspace({child_obj})" in state:
                # Increase the likelihood of sampling outside the workspace.
                r = BeyondWorkspace.WORKSPACE_RADIUS
                xyz_min[0] = r * np.cos(np.arcsin(0.5 * (xyz_max[1] - xyz_min[1]) / r))
                xyz_max[0] -= 0.05
                xyz_min[1] += 0.05
                xyz_max[1] -= 0.05
            elif f"inworkspace({child_obj})" in state:
                # Increase the likelihood of sampling inside the workspace.
                xyz_min[0] = InWorkspace.MIN_X
                xyz_max[0] = BeyondWorkspace.WORKSPACE_RADIUS
                xyz_min[1] += 0.05
                xyz_max[1] -= 0.05

        # Generate pose on parent.
        xyz = np.zeros(3)
        xyz[:2] = np.random.uniform(xyz_min[:2], xyz_max[:2])
        xyz[2] = xyz_max[2] + 0.5 * child_obj.size[2] + 0.01
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        quat = eigen.Quaterniond(aa)

        # Tip the object over.
        if f"istippable({child_obj})" in state and not child_obj.isinstance(Hook):
            if np.random.random() < self.TIPPED_PROB:
                axis = np.random.uniform(-1, 1, size=2)
                axis /= np.linalg.norm(axis)
                quat = quat * eigen.Quaterniond(
                    eigen.AngleAxisd(np.pi / 2, np.array([*axis, 0.0]))
                )
                xyz[2] = xyz_max[2] + 0.8 * child_obj.size[:2].max()

        pose = math.Pose(pos=xyz, quat=quat.coeffs)

        child_obj.set_pose(pose)

        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        child_obj, parent_obj = self._get_arg_objects(objects)
        if child_obj.isinstance(Null):
            return True

        if not is_above(child_obj.aabb(), parent_obj.aabb()):
            return False

        if f"istippable({child_obj})" not in state or child_obj.isinstance(Hook):
            child_pose = child_obj.pose()
            if not is_upright(child_pose.quat):
                return False

        return True


def generate_grasp_pose(obj: Object, handlegrasp: bool = False) -> math.Pose:
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

    return math.Pose(pos=xyz, quat=eigen.Quaterniond(aa))


class HandleGrasp(Predicate):
    pass


class Inhand(Predicate):
    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        from temporal_policies.envs.pybullet.table.primitives import compute_lift_height

        obj = self._get_arg_objects(objects)[0]
        if obj.is_static:
            return True

        obj.disable_collisions()

        # Generate grasp pose.
        grasp_pose = generate_grasp_pose(obj, f"handlegrasp({obj})" in state)
        obj_pose = math.Pose.from_eigen(grasp_pose.to_eigen().inverse())
        obj_pose.pos += robot.home_pose.pos

        # Generate post-pick pose.
        table_xyz_min, table_xyz_max = objects["table"].aabb()
        table_xyz_min[0] = InWorkspace.MIN_X
        xyz_pick = np.array([0.0, 0.0, compute_lift_height(obj.size)])
        for _ in range(100):
            xyz_pick[:2] = np.random.uniform(table_xyz_min[:2], table_xyz_max[:2])
            if np.linalg.norm(xyz_pick[:2]) <= BeyondWorkspace.WORKSPACE_RADIUS:
                break
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))

        # Use fake grasp.
        obj.set_pose(obj_pose)
        robot.grasp_object(obj, realistic=False)
        try:
            robot.goto_pose(pos=xyz_pick, quat=eigen.Quaterniond(aa))
        except ControlException:
            robot.reset()
            return False

        obj.enable_collisions()

        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        return True
        # obj = self._get_arg_objects(objects)[0]
        #
        # xyz_min, xyz_max = obj.aabb()
        #
        # xyz_ee = robot.arm.ee_pose().pos
        #
        # if (xyz_min > xyz_ee).any() or (xyz_ee > xyz_max).any():
        #     return False
        #
        # return True
