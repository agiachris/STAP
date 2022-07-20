import abc
from typing import Dict, List

from ctrlutils import eigen
import gym
import numpy as np
import symbolic

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet.sim import robot
from temporal_policies.envs.pybullet.sim.robot import ControlException
from temporal_policies.envs.pybullet.table import objects


def compute_top_down_orientation(
    quat_home: eigen.Quaterniond, quat_obj: eigen.Quaterniond, theta: float
) -> eigen.Quaterniond:
    quat_ee_to_obj = quat_home
    quat_obj_to_world = eigen.AngleAxisd(quat_obj)
    command_aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    command_quat = quat_obj_to_world * command_aa * quat_ee_to_obj
    return command_quat


def is_upright(quat: np.ndarray) -> bool:
    aa = eigen.AngleAxisd(eigen.Quaterniond(quat))
    return abs(aa.axis.dot(np.array([0.0, 0.0, 1.0]))) >= 0.99


class Primitive(envs.Primitive, abc.ABC):
    @abc.abstractmethod
    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        pass

    @abc.abstractmethod
    def sample_action(self) -> np.ndarray:
        pass

    @staticmethod
    def from_action_call(
        action_call: str,
        primitives: List[str],
        objects: Dict[str, objects.Object],
    ) -> "Primitive":
        name, arg_names = symbolic.parse_proposition(action_call)
        args = [objects[obj_name] for obj_name in arg_names]

        primitive_class = globals()[name.capitalize()]
        idx_policy = primitives.index(name)
        return primitive_class(idx_policy, args)

    # def __repr__(self) -> str:
    #     return f"{type(self).__name__.lower()}({', '.join([arg.name for arg in self.policy_args])})"


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
        obj = self.policy_args[0]
        obj_pose = obj.pose()
        obj_quat = eigen.Quaterniond(obj_pose.quat)

        # Compute position.
        command_pos = obj_pose.pos + obj_quat * pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(
            eigen.Quaterniond(robot.home_pose.quat), obj_quat, theta
        )

        pre_pos = np.append(command_pos[:2], obj.aabb()[1, 2] + 0.1)
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos, command_quat)
            if not robot.grasp_object(obj):
                # print(f"Robot.grasp_object({obj}) failed")
                return False
            robot.goto_pose(pre_pos, command_quat)
        except ControlException:
            return False

        return True

    def sample_action(self) -> np.ndarray:
        if self.policy_args[0].isinstance(objects.Hook):
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
        target = self.policy_args[1]
        target_pose = target.pose()
        target_quat = eigen.Quaterniond(target_pose.quat)

        # Compute position.
        command_pos = target_pose.pos + target_quat * pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(
            eigen.Quaterniond(robot.home_pose.quat), target_quat, theta
        )

        pre_pos = np.append(command_pos[:2], target.aabb()[1, 2] + 0.1)
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos, command_quat)
            robot.grasp(0)
            robot.goto_pose(pre_pos, command_quat)
        except ControlException:
            return False

        return True

    def sample_action(self) -> np.ndarray:
        return np.array([0.4, 0.0, 0.04, 0.0], dtype=np.float32)


class Pull(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(
        low=np.array([-0.4, 0.0, -0.5, -np.pi], dtype=np.float32),
        high=np.array([0.0, 0.4, 0.5, np.pi], dtype=np.float32),
    )

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        PULL_HEIGHT = 0.02
        MIN_PULL_DISTANCE = 0.01

        # Parse action.
        r_reach, r_pull, y, theta = self.scale_action(action)

        # Get target pose in polar coordinates
        target, hook = self.policy_args
        target_pose = target.pose()
        if target_pose.pos[0] < 0:
            return False

        target_distance = np.linalg.norm(target_pose.pos[:2])
        target_xy = target_pose.pos[:2] / target_distance
        target_theta = np.arctan2(target_xy[1], target_xy[0])
        target_aa = eigen.AngleAxisd(target_theta, np.array([0.0, 0.0, 1.0]))
        target_quat = eigen.Quaterniond(target_aa)

        target_pos = np.append(target_pose.pos[:2], PULL_HEIGHT)

        # Compute position.
        pos_reach = np.array([r_reach, y, 0.0])
        command_pos_reach = target_pos + target_quat * pos_reach
        pos_pull = np.array([r_reach - r_pull, y, 0.0])
        command_pos_pull = target_pos + target_quat * pos_pull

        # Compute orientation.
        command_quat = compute_top_down_orientation(
            eigen.Quaterniond(robot.home_pose.quat), target_quat, theta
        )

        pre_pos = np.append(command_pos_reach[:2], target.aabb()[1, 2] + 0.1)
        post_pos = np.append(command_pos_pull[:2], target.aabb()[1, 2] + 0.1)
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos_reach, command_quat)
            if not is_upright(target.pose().quat):
                return False
            robot.goto_pose(command_pos_pull, command_quat)
            if not is_upright(target.pose().quat):
                return False
            robot.goto_pose(post_pos, command_quat)
        except ControlException:
            return False

        new_target_pose = target.pose()
        new_target_distance = np.linalg.norm(new_target_pose.pos[:2])
        if new_target_distance >= target_distance - MIN_PULL_DISTANCE:
            return False

        return True

    def sample_action(self) -> np.ndarray:
        return np.array([-0.1, 0.2, -0.1, 0.0], dtype=np.float32)
