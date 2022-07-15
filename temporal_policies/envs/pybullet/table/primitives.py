import abc
from typing import Dict, List

from ctrlutils import eigen
import gym
import numpy as np
import symbolic

from temporal_policies.envs.pybullet.sim import robot
from temporal_policies.envs.pybullet.sim.robot import ControlException
from temporal_policies.envs.pybullet.table import objects
from temporal_policies.utils import spaces


def compute_top_down_orientation(
    quat_home: eigen.Quaterniond, quat_obj: eigen.Quaterniond, theta: float
) -> eigen.Quaterniond:
    quat_ee_to_obj = quat_home
    quat_obj_to_world = eigen.AngleAxisd(quat_obj)
    command_aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    command_quat = quat_obj_to_world * command_aa * quat_ee_to_obj
    return command_quat


class Primitive(abc.ABC):
    action_space: gym.spaces.Box
    action_scale: gym.spaces.Box

    def __init__(self, args: List[objects.Object]):
        self._args = args

    @property
    def args(self) -> List[objects.Object]:
        return self._args

    @abc.abstractmethod
    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        pass

    @staticmethod
    def from_action_call(action_call: str, objects: Dict[str, objects.Object]) -> "Primitive":
        name, arg_names = symbolic.parse_proposition(action_call)
        args = [objects[obj_name] for obj_name in arg_names]

        primitives = {
            "pick": Pick,
            "place": Place,
        }

        return primitives[name](args)

    @classmethod
    def scale_action(cls, action: np.ndarray) -> np.ndarray:
        return spaces.transform(
            action, from_space=cls.action_space, to_space=cls.action_scale
        )

    def normalize_action(cls, action: np.ndarray) -> np.ndarray:
        return spaces.transform(
            action, from_space=cls.action_scale, to_space=cls.action_space
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__.lower()}({', '.join([arg.name for arg in self.args])})"

    @abc.abstractmethod
    def sample_action(self) -> np.ndarray:
        pass


class Pick(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(
        low=np.array([-0.1, -0.1, -0.05, -np.pi], dtype=np.float32),
        high=np.array([0.2, 0.1, 0.05, np.pi], dtype=np.float32),
    )

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        # Parse action.
        action = self.scale_action(action)
        pos = action[:3]
        theta = action[3]

        # Get object pose.
        obj = self.args[0]
        obj_pose = obj.pose()
        obj_quat = eigen.Quaterniond(obj_pose.quat)

        # Compute position.
        command_pos = obj_pose.pos + obj_quat * pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(
            eigen.Quaterniond(robot.home_pose.quat), obj_quat, theta
        )

        pre_pos = np.array([*command_pos[:2], obj.aabb()[1, 2] + 0.1])
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos, command_quat)
            if not robot.grasp_object(obj):
                # print(f"Robot.grasp_object({obj}) failed")
                return False
            robot.goto_pose(pre_pos, command_quat)
        except ControlException as e:
            # print(f"ControlException: {e}")
            return False

        return True

    def sample_action(self) -> np.ndarray:
        if self.args[0].isinstance(objects.Hook):
            return np.array([-0.1, -0.1, 0.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class Place(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(
        low=np.array([-0.3, -0.5, -0.05, -np.pi], dtype=np.float32),
        high=np.array([0.9, 0.5, 0.1, np.pi], dtype=np.float32),
    )

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        # Parse action.
        action = self.scale_action(action)
        pos = action[:3]
        theta = action[3]

        # Get target pose.
        target = self.args[1]
        target_pose = target.pose()
        target_quat = eigen.Quaterniond(target_pose.quat)

        # Compute position.
        command_pos = target_pose.pos + target_quat * pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(
            eigen.Quaterniond(robot.home_pose.quat), target_quat, theta
        )

        pre_pos = np.array([*command_pos[:2], target.aabb()[1, 2] + 0.1])
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos, command_quat)
            robot.grasp(0)
            robot.goto_pose(pre_pos, command_quat)
        except ControlException as e:
            # print(f"ControlException: {e}")
            return False

        return True

    def sample_action(self) -> np.ndarray:
        return np.array([0.4, 0.0, 0.04, 0.0], dtype=np.float32)
