import dataclasses
from typing import Dict, List

from ctrlutils import eigen
import numpy as np
import symbolic

from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.envs.pybullet.sim import math
from temporal_policies.envs.pybullet.sim.robot import ControlException, Robot


@dataclasses.dataclass
class Proposition:
    args: List[str]

    @classmethod
    def create(cls, proposition: str) -> "Proposition":
        predicate, args = symbolic.parse_proposition(proposition)
        predicate_class = globals()[predicate.capitalize()]
        return predicate_class(args)

    def sample(self, robot: Robot, objects: Dict[str, Object]) -> None:
        raise NotImplementedError

    def value(self, robot: Robot, objects: Dict[str, Object]) -> bool:
        raise NotImplementedError

    def _get_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        return [objects[arg] for arg in self.args]


class On(Proposition):
    def sample(self, robot: Robot, objects: Dict[str, Object]) -> bool:
        child_obj, parent_obj = self._get_arg_objects(objects)
        if child_obj.is_static:
            return True

        # Generate pose on parent.
        xyz_min, xyz_max = parent_obj.aabb()
        xyz = np.zeros(3)
        xyz[:2] = np.random.uniform(0.9 * xyz_min[:2], 0.9 * xyz_max[:2])
        xyz[2] = xyz_max[2] + 0.1
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        pose = math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs)

        child_obj.set_pose(pose)

        return True

    def value(self, robot: Robot, objects: Dict[str, Object]) -> bool:
        child_obj, parent_obj = self._get_arg_objects(objects)

        if child_obj.aabb()[0, 2] < parent_obj.aabb()[1, 2] - 0.01:
            return False

        child_pose = child_obj.pose()
        child_aa = eigen.AngleAxisd(eigen.Quaterniond(child_pose.quat))
        if abs(child_aa.axis.dot(np.array([0.0, 0.0, 1.0]))) < 0.99:
            return False

        return True


class Inhand(Proposition):
    def sample(self, robot: Robot, objects: Dict[str, Object]) -> bool:
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

    def value(self, robot: Robot, objects: Dict[str, Object]) -> bool:
        return True
        obj = self._get_arg_objects(objects)[0]

        xyz_min, xyz_max = obj.aabb()

        xyz_ee = robot.arm.ee_pose().pos

        if (xyz_min > xyz_ee).any() or (xyz_ee > xyz_max).any():
            return False

        return True
