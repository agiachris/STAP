import abc
from typing import Dict, List, Type

from ctrlutils import eigen
import gym
import numpy as np
import symbolic

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet.sim import robot
from temporal_policies.envs.pybullet.sim.robot import ControlException
from temporal_policies.envs.pybullet.table import objects, primitive_actions

dbprint = lambda *args: None  # noqa
# dbprint = print


def compute_top_down_orientation(quat_obj: eigen.Quaterniond, theta: float) -> eigen.Quaterniond:
    quat_obj_to_world = eigen.AngleAxisd(quat_obj)
    command_aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    command_quat = quat_obj_to_world * command_aa
    return command_quat


def is_upright(quat: np.ndarray) -> bool:
    aa = eigen.AngleAxisd(eigen.Quaterniond(quat))
    return abs(aa.axis.dot(np.array([0.0, 0.0, 1.0]))) >= 0.99


class Primitive(envs.Primitive, abc.ABC):
    Action: Type[primitive_actions.PrimitiveAction]

    @abc.abstractmethod
    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        pass

    @abc.abstractmethod
    def sample_action(self) -> primitive_actions.PrimitiveAction:
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


class Pick(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PickAction.range())
    Action = primitive_actions.PickAction

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        # Parse action.
        a = primitive_actions.PickAction(self.scale_action(action))
        dbprint(a)

        # Get object pose.
        obj = self.policy_args[0]
        obj_pose = obj.pose()
        obj_quat = eigen.Quaterniond(obj_pose.quat)

        # Compute position.
        command_pos = obj_pose.pos + obj_quat * a.pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(obj_quat, a.theta.item())

        pre_pos = np.append(command_pos[:2], obj.aabb()[1, 2] + 0.1)
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos, command_quat)
            if not robot.grasp_object(obj):
                dbprint(f"Robot.grasp_object({obj}) failed")
                return False
            robot.goto_pose(pre_pos, command_quat)
        except ControlException:
            return False

        return True

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        if self.policy_args[0].isinstance(objects.Hook):
            pos = np.array([-0.1, -0.1, 0.0])
        else:
            pos = np.array([0.0, 0.0, 0.0])
        return primitive_actions.PickAction(pos=pos, theta=0.0)


class Place(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PlaceAction.range())
    Action = primitive_actions.PlaceAction

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        # Parse action.
        a = primitive_actions.PlaceAction(self.scale_action(action))
        dbprint(a)

        # Get target pose.
        target = self.policy_args[1]
        target_pose = target.pose()
        target_quat = eigen.Quaterniond(target_pose.quat)

        # Compute position.
        command_pos = target_pose.pos + target_quat * a.pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(target_quat, a.theta.item())

        pre_pos = np.append(command_pos[:2], target.aabb()[1, 2] + 0.1)
        try:
            robot.goto_pose(pre_pos, command_quat)
            robot.goto_pose(command_pos, command_quat)
            robot.grasp(0)
            robot.goto_pose(pre_pos, command_quat)
        except ControlException:
            return False

        return True

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PlaceAction(
            pos=np.array([0.4, 0.0, self.policy_args[0].size[2]]), theta=0.0
        )

    @classmethod
    def action(cls, action: np.ndarray) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PlaceAction(action)


class Pull(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PullAction.range())
    Action = primitive_actions.PullAction

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        PULL_HEIGHT = 0.03
        MIN_PULL_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PullAction(self.scale_action(action))
        dbprint(a)

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
        pos_reach = np.array([a.r_reach, a.y, 0.0])
        command_pos_reach = target_pos + target_quat * pos_reach
        pos_pull = np.array([a.r_reach - a.r_pull, a.y, 0.0])
        command_pos_pull = target_pos + target_quat * pos_pull

        # Compute orientation.
        command_quat = compute_top_down_orientation(target_quat, a.theta.item())

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

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PullAction(r_reach=-0.1, r_pull=0.2, y=-0.1, theta=0.0)
