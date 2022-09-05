import abc
import random
from typing import Callable, Dict, List, NamedTuple, Type

from ctrlutils import eigen
import gym
import numpy as np
import symbolic

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet.sim import robot, math
from temporal_policies.envs.pybullet.sim.robot import ControlException
from temporal_policies.envs.pybullet.table.objects import Box, Hook, Rack, Null, Object
from temporal_policies.envs.pybullet.table import (
    object_state,
    utils,
    primitive_actions,
)


dbprint = lambda *args: None  # noqa
# dbprint = print


ACTION_CONSTRAINTS = {"max_lift_height": 0.4, "max_lift_radius": 0.7}


def compute_top_down_orientation(
    theta: float, quat_obj: eigen.Quaterniond = eigen.Quaterniond.identity()
) -> eigen.Quaterniond:
    """Computes the top-down orientation of the end-effector with respect to a target object.

    Args:
        theta: Angle of the gripper about the world z-axis wrt the target object.
        quat_obj: Orientation of the target object.
    """
    command_aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    command_quat = quat_obj * eigen.Quaterniond(command_aa)
    return command_quat


def did_object_move(
    obj: Object,
    old_pose: math.Pose,
    max_delta_xyz: float = 0.05,
    max_delta_theta: float = 5.0 * np.pi / 180,
) -> bool:
    """Checks if the object has moved significantly from its old pose."""
    new_pose = obj.pose()
    T_old_to_world = old_pose.to_eigen()
    T_new_to_world = new_pose.to_eigen()
    T_new_to_old = T_old_to_world.inverse() * T_new_to_world

    delta_xyz = float(np.linalg.norm(T_new_to_old.translation))
    delta_theta = eigen.AngleAxisd(eigen.Quaterniond(T_new_to_old.linear)).angle

    return delta_xyz >= max_delta_xyz or delta_theta >= max_delta_theta


def initialize_robot_pose(robot: robot.Robot) -> bool:
    x_min, x_max = (
        utils.TABLE_CONSTRAINTS["workspace_x_min"],
        ACTION_CONSTRAINTS["max_lift_radius"],
    )
    y_min, y_max = primitive_actions.PlaceAction.RANGES["y"]
    xy_min = np.array([x_min, y_min])
    xy_max = np.array([x_max, y_max])

    while True:
        xy = np.random.uniform(xy_min, xy_max)
        if np.linalg.norm(xy) < ACTION_CONSTRAINTS["max_lift_radius"]:
            break
    theta = np.random.uniform(*object_state.ObjectState.RANGES["wz"])

    pos = np.append(xy, ACTION_CONSTRAINTS["max_lift_height"])
    aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
    quat = eigen.Quaterniond(aa)

    try:
        robot.goto_pose(pos, quat)
    except ControlException as e:
        dbprint("initialize_robot_pose():\n", e)
        return False

    return True


class ExecutionResult(NamedTuple):
    """Return tuple from Primitive.execute()."""

    success: bool
    """Whether the action succeeded."""

    truncated: bool
    """Whether the action was truncated because of a control error."""


class Primitive(envs.Primitive, abc.ABC):
    Action: Type[primitive_actions.PrimitiveAction]

    @abc.abstractmethod
    def execute(
        self,
        action: np.ndarray,
        robot: robot.Robot,
        objects: Dict[str, Object],
        wait_until_stable_fn: Callable[[], int],
    ) -> ExecutionResult:
        """Executes the primitive.

        Args:
            action: Normalized action (inside action_space, not action_scale).
            robot: Robot to control.
            objects: List of objects in the environment.
            wait_until_stable_fn: Function to run sim until environment is stable.
        Returns:
            (success, truncated) 2-tuple.
        """

    def sample(self) -> np.ndarray:
        if random.random() < 0.9:
            action = self.normalize_action(self.sample_action().vector)
            action = np.random.normal(loc=action, scale=0.05)
            action = action.astype(np.float32).clip(
                self.action_space.low, self.action_space.high
            )
            return action
        else:
            return super().sample()

    @abc.abstractmethod
    def sample_action(self) -> primitive_actions.PrimitiveAction:
        pass

    def get_non_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        """Gets the non-primitive argument objects."""
        return [
            obj
            for obj in objects.values()
            if obj not in self.policy_args and not obj.isinstance(Null)
        ]

    def create_non_arg_movement_check(
        self, objects: Dict[str, Object]
    ) -> Callable[[], bool]:
        """Returns a function that checks if any non-primitive argument has been significantly perturbed."""
        # Get non-arg object poses.
        non_arg_objects = self.get_non_arg_objects(objects)
        old_poses = [obj.pose() for obj in non_arg_objects]

        def did_non_args_move() -> bool:
            """Checks if any object has moved significantly from its old pose."""
            return any(
                did_object_move(obj, old_pose)
                for obj, old_pose in zip(non_arg_objects, old_poses)
            )

        return did_non_args_move

    @staticmethod
    def from_action_call(
        action_call: str,
        primitives: List[str],
        objects: Dict[str, Object],
    ) -> "Primitive":
        name, arg_names = symbolic.parse_proposition(action_call)
        args = [objects[obj_name] for obj_name in arg_names]

        primitive_class = globals()[name.capitalize()]
        idx_policy = primitives.index(name)
        return primitive_class(idx_policy, args)

    def __eq__(self, other) -> bool:
        if isinstance(other, Primitive):
            return str(self) == str(other)
        else:
            return False


class Pick(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PickAction.range())
    Action = primitive_actions.PickAction
    ALLOW_COLLISIONS = False

    def execute(
        self,
        action: np.ndarray,
        robot: robot.Robot,
        objects: Dict[str, Object],
        wait_until_stable_fn: Callable[[], int],
    ) -> ExecutionResult:
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
        command_quat = compute_top_down_orientation(a.theta.item(), obj_quat)

        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        did_non_args_move = self.create_non_arg_movement_check(objects)
        try:
            robot.goto_pose(pre_pos, command_quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_quat}) collided"
                )

            robot.goto_pose(
                command_pos,
                command_quat,
                check_collisions=[
                    obj.body_id for obj in self.get_non_arg_objects(objects)
                ],
            )

            if not robot.grasp_object(obj):
                raise ControlException(f"Robot.grasp_object({obj}) failed")

            robot.goto_pose(pre_pos, command_quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_quat}) collided"
                )
        except ControlException as e:
            dbprint("Pick.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        obj = self.policy_args[0]
        if obj.isinstance(Hook):
            pos_handle, pos_head, _ = Hook.compute_link_positions(
                obj.head_length, obj.handle_length, obj.handle_y, obj.radius
            )
            action_range = self.Action.range()
            if random.random() < obj.handle_length / (
                obj.handle_length + obj.head_length
            ):
                # Handle.
                random_x = np.random.uniform(*action_range[:, 0])
                pos = np.array([random_x, pos_handle[1], 0])
                theta = 0.0
            else:
                # Head.
                random_y = np.random.uniform(*action_range[:, 1])
                pos = np.array([pos_head[0], random_y, 0])
                theta = np.pi / 2
        elif obj.isinstance(Box):
            pos = np.array([0.0, 0.0, 0.0])
            theta = 0.0  # if random.random() <= 0.5 else np.pi / 2
        else:
            pos = np.array([0.0, 0.0, 0.0])
            theta = 0.0
        return primitive_actions.PickAction(pos=pos, theta=theta)


class Place(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PlaceAction.range())
    Action = primitive_actions.PlaceAction
    ALLOW_COLLISIONS = False

    def execute(
        self,
        action: np.ndarray,
        robot: robot.Robot,
        objects: Dict[str, Object],
        wait_until_stable_fn: Callable[[], int],
    ) -> ExecutionResult:
        MAX_DROP_DISTANCE = 0.05

        # Parse action.
        a = primitive_actions.PlaceAction(self.scale_action(action))
        dbprint(a)

        obj: Object = self.policy_args[0]  # type: ignore
        target: Object = self.policy_args[1]  # type: ignore

        # Get target pose.
        target_pose = target.pose()
        target_quat = eigen.Quaterniond(target_pose.quat)

        # Compute position.
        command_pos = target_pose.pos + target_quat * a.pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(a.theta.item(), target_quat)

        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        did_non_args_move = self.create_non_arg_movement_check(objects)
        try:
            robot.goto_pose(pre_pos, command_quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_quat}) collided"
                )

            robot.goto_pose(
                command_pos,
                command_quat,
                check_collisions=[target.body_id]
                + [obj.body_id for obj in self.get_non_arg_objects(objects)],
            )

            # Make sure object won't drop from too high.
            if not utils.is_within_distance(
                obj, target, MAX_DROP_DISTANCE, robot.physics_id
            ):
                raise ControlException("Object dropped from too high.")

            robot.grasp(0)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException("Robot.grasp(0) collided")

            robot.goto_pose(pre_pos, command_quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_quat}) collided"
                )
        except ControlException as e:
            # If robot fails before grasp(0), object may still be grasped.
            dbprint("Place.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        wait_until_stable_fn()

        if utils.is_below_table(obj):
            # Falling off the table is an exception.
            return ExecutionResult(success=False, truncated=True)

        if not utils.is_upright(obj) or not utils.is_above(obj, target):
            return ExecutionResult(success=False, truncated=False)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()
        action_range = action.range()

        # Generate a random xy in the aabb of the parent.
        parent = self.policy_args[1]
        xy_min = np.maximum(action_range[0, :2], -0.5 * parent.size[:2])
        xy_max = np.minimum(action_range[1, :2], 0.5 * parent.size[:2])
        action.pos[:2] = np.random.uniform(xy_min, xy_max)

        # Compute an appropriate place height given the grasped object's height.
        obj = self.policy_args[0]
        z_gripper = ACTION_CONSTRAINTS["max_lift_height"]
        z_obj = obj.pose().pos[2]
        action.pos[2] = z_gripper - z_obj + 0.5 * obj.size[2]

        action.pos[2] = np.clip(action.pos[2], action_range[0, 2], action_range[1, 2])

        return action

    @classmethod
    def action(cls, action: np.ndarray) -> primitive_actions.PrimitiveAction:
        return primitive_actions.PlaceAction(action)


class Pull(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PullAction.range())
    Action = primitive_actions.PullAction
    ALLOW_COLLISIONS = True

    def execute(
        self,
        action: np.ndarray,
        robot: robot.Robot,
        objects: Dict[str, Object],
        wait_until_stable_fn: Callable[[], int],
    ) -> ExecutionResult:
        PULL_HEIGHT = 0.03
        MIN_PULL_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PullAction(self.scale_action(action))
        dbprint(a)

        # Get target pose in polar coordinates
        target, hook = self.policy_args
        target_pose = target.pose()
        if target_pose.pos[0] < 0:
            return ExecutionResult(success=False, truncated=True)

        target_distance = np.linalg.norm(target_pose.pos[:2])
        reach_xy = target_pose.pos[:2] / target_distance
        reach_theta = np.arctan2(reach_xy[1], reach_xy[0])
        reach_aa = eigen.AngleAxisd(reach_theta, np.array([0.0, 0.0, 1.0]))
        reach_quat = eigen.Quaterniond(reach_aa)

        target_pos = np.append(target_pose.pos[:2], PULL_HEIGHT)
        T_hook_to_world = hook.pose().to_eigen()
        T_gripper_to_world = robot.arm.ee_pose().to_eigen()
        T_gripper_to_hook = T_hook_to_world.inverse() * T_gripper_to_world

        # Compute position.
        pos_reach = np.array([a.r_reach, a.y, 0.0])
        hook_pos_reach = target_pos + reach_quat * pos_reach
        pos_pull = np.array([a.r_reach + a.r_pull, a.y, 0.0])
        hook_pos_pull = target_pos + reach_quat * pos_pull

        # Compute orientation.
        hook_quat = compute_top_down_orientation(a.theta.item(), reach_quat)

        T_reach_hook_to_world = math.Pose(hook_pos_reach, hook_quat).to_eigen()
        T_pull_hook_to_world = math.Pose(hook_pos_pull, hook_quat).to_eigen()
        T_reach_to_world = T_reach_hook_to_world * T_gripper_to_hook
        T_pull_to_world = T_pull_hook_to_world * T_gripper_to_hook
        command_pose_reach = math.Pose.from_eigen(T_reach_to_world)
        command_pose_pull = math.Pose.from_eigen(T_pull_to_world)

        pre_pos = np.append(
            command_pose_reach.pos[:2], ACTION_CONSTRAINTS["max_lift_height"]
        )
        post_pos = np.append(
            command_pose_pull.pos[:2], ACTION_CONSTRAINTS["max_lift_height"]
        )

        did_non_args_move = self.create_non_arg_movement_check(objects)
        try:
            robot.goto_pose(pre_pos, command_pose_reach.quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_pose_reach.quat}) collided"
                )

            robot.goto_pose(
                command_pose_reach.pos,
                command_pose_reach.quat,
                check_collisions=[
                    obj.body_id for obj in self.get_non_arg_objects(objects)
                ],
            )
            if not utils.is_upright(target):
                raise ControlException("Target is not upright", target.pose().quat)

            robot.goto_pose(
                command_pose_pull.pos,
                command_pose_pull.quat,
                pos_gains=np.array([[49, 14], [49, 14], [121, 22]]),
            )
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({command_pose_pull.pos}, {command_pose_pull.quat}) collided"
                )

            robot.goto_pose(post_pos, command_pose_pull.quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({post_pos}, {command_pose_pull.quat}) collided"
                )
        except ControlException as e:
            dbprint("Pull.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        wait_until_stable_fn()

        if not utils.is_upright(target):
            return ExecutionResult(success=False, truncated=False)

        new_target_distance = np.linalg.norm(target.pose().pos[:2])
        if new_target_distance >= target_distance - MIN_PULL_DISTANCE:
            return ExecutionResult(success=False, truncated=False)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()

        obj, hook = self.policy_args
        obj_halfsize = 0.5 * np.linalg.norm(obj.size[:2])
        collision_length = 0.5 * hook.size[0] - 2 * hook.radius - obj_halfsize
        action.r_reach = -collision_length
        action.theta = 0.125 * np.pi

        return action


class Push(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PushAction.range())
    Action = primitive_actions.PushAction
    ALLOW_COLLISIONS = True

    def execute(
        self,
        action: np.ndarray,
        robot: robot.Robot,
        objects: Dict[str, Object],
        wait_until_stable_fn: Callable[[], int],
    ) -> ExecutionResult:
        PUSH_HEIGHT = 0.03
        MIN_PUSH_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PushAction(self.scale_action(action))
        dbprint(a)

        # Get target pose in polar coordinates
        target, hook = self.policy_args
        target_pose = target.pose()
        if target_pose.pos[0] < 0:
            return ExecutionResult(success=False, truncated=True)

        target_distance = np.linalg.norm(target_pose.pos[:2])
        reach_xy = target_pose.pos[:2] / target_distance
        reach_theta = np.arctan2(reach_xy[1], reach_xy[0])
        reach_aa = eigen.AngleAxisd(reach_theta, np.array([0.0, 0.0, 1.0]))
        reach_quat = eigen.Quaterniond(reach_aa)

        target_pos = np.append(target_pose.pos[:2], PUSH_HEIGHT)
        T_hook_to_world = hook.pose().to_eigen()
        T_gripper_to_world = robot.arm.ee_pose().to_eigen()
        T_gripper_to_hook = T_hook_to_world.inverse() * T_gripper_to_world

        # Compute position.
        pos_reach = np.array([a.r_reach, a.y, 0.0])
        hook_pos_reach = target_pos + reach_quat * pos_reach
        pos_push = np.array([a.r_reach + a.r_push, a.y, 0.0])
        hook_pos_push = target_pos + reach_quat * pos_push

        # Compute orientation.
        hook_quat = compute_top_down_orientation(a.theta.item(), reach_quat)

        T_reach_hook_to_world = math.Pose(hook_pos_reach, hook_quat).to_eigen()
        T_push_hook_to_world = math.Pose(hook_pos_push, hook_quat).to_eigen()
        T_reach_to_world = T_reach_hook_to_world * T_gripper_to_hook
        T_push_to_world = T_push_hook_to_world * T_gripper_to_hook
        command_pose_reach = math.Pose.from_eigen(T_reach_to_world)
        command_pose_push = math.Pose.from_eigen(T_push_to_world)
        pre_pos = np.append(
            command_pose_reach.pos[:2], ACTION_CONSTRAINTS["max_lift_height"]
        )

        did_non_args_move = self.create_non_arg_movement_check(objects)
        try:
            robot.goto_pose(pre_pos, command_pose_reach.quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_pose_reach.quat}) collided"
                )

            robot.goto_pose(
                command_pose_reach.pos,
                command_pose_reach.quat,
                check_collisions=[
                    obj.body_id for obj in self.get_non_arg_objects(objects)
                ],
            )
            if not utils.is_upright(target):
                raise ControlException("Target is not upright", target.pose().quat)

            robot.goto_pose(
                command_pose_push.pos,
                command_pose_push.quat,
                pos_gains=np.array([[49, 14], [49, 14], [121, 22]]),
            )
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({command_pose_push.pos}, {command_pose_push.quat}) collided"
                )

            robot.goto_pose(
                command_pose_reach.pos,
                command_pose_reach.quat,
                check_collisions=[
                    obj.body_id for obj in self.get_non_arg_objects(objects)
                ],
            )
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({command_pose_reach.pos}, {command_pose_reach.quat}) collided"
                )

            robot.goto_pose(pre_pos, command_pose_reach.quat)
            if not self.ALLOW_COLLISIONS and did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({pre_pos}, {command_pose_reach.quat}) collided"
                )

        except ControlException as e:
            dbprint("Push.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        wait_until_stable_fn()

        if not utils.is_upright(target):
            return ExecutionResult(success=False, truncated=False)

        new_target_distance = np.linalg.norm(target.pose().pos[:2])
        if new_target_distance <= target_distance + MIN_PUSH_DISTANCE:
            return ExecutionResult(success=False, truncated=False)

        # Target must be pushed underneath rack if it exists
        for obj in self.get_non_arg_objects(objects):
            if obj.isinstance(Rack):
                if not utils.is_under(target, obj):
                    return ExecutionResult(success=False, truncated=False)
                break

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()

        obj, hook = self.policy_args
        obj_halfsize = 0.5 * np.linalg.norm(obj.size[:2])
        collision_length = -0.5 * hook.size[0] - 2 * hook.radius - obj_halfsize
        action.r_reach = collision_length
        action.theta = 0.125 * np.pi

        return action
