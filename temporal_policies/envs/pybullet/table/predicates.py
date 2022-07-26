import dataclasses
from typing import Dict, List, Sequence

from ctrlutils import eigen
import numpy as np
import symbolic

from temporal_policies.envs.pybullet.table.objects import Hook, Object
from temporal_policies.envs.pybullet.table.primitives import is_upright
from temporal_policies.envs.pybullet.sim import math
from temporal_policies.envs.pybullet.sim.robot import ControlException, Robot


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
        return bool(np.linalg.norm(obj.pose().pos[:2]) > self.WORKSPACE_RADIUS)


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
        if parent_obj.name == "table" and f"beyondworkspace({child_obj})" in state:
            # Increase the likelihood of sampling outside the workspace.
            xyz_min[0] = BeyondWorkspace.WORKSPACE_RADIUS

        # Generate pose on parent.
        xyz = np.zeros(3)
        xyz[:2] = np.random.uniform(0.9 * xyz_min[:2], 0.9 * xyz_max[:2])
        xyz[2] = xyz_max[2] + 0.6 * child_obj.size[2]
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

        if child_obj.aabb()[0, 2] < parent_obj.aabb()[1, 2] - 0.01:
            return False

        if f"istippable({child_obj})" not in state or child_obj.isinstance(Hook):
            child_pose = child_obj.pose()
            if not is_upright(child_pose.quat):
                return False

        return True


class Inhand(Predicate):
    def sample(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        obj = self._get_arg_objects(objects)[0]
        if obj.is_static:
            return True

        obj.disable_collisions()

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
        obj.set_pose(pose)
        robot.grasp_object(obj, realistic=False)
        try:
            robot.goto_pose(pos=xyz_pick)
        except ControlException:
            robot.reset()
            return False

        obj.enable_collisions()

        return True

    def value(
        self, robot: Robot, objects: Dict[str, Object], state: Sequence[Predicate]
    ) -> bool:
        return True
        obj = self._get_arg_objects(objects)[0]

        xyz_min, xyz_max = obj.aabb()

        xyz_ee = robot.arm.ee_pose().pos

        if (xyz_min > xyz_ee).any() or (xyz_ee > xyz_max).any():
            return False

        return True
