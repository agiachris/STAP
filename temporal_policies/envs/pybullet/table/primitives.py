import abc
import random
from typing import Callable, Dict, List, Optional, NamedTuple, Type

from ctrlutils import eigen
import gym
import numpy as np
import symbolic

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet.sim import math
from temporal_policies.envs.pybullet.sim.robot import ControlException, Robot
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


def initialize_robot_pose(robot: Robot) -> bool:
    x_min, x_max = (
        utils.TABLE_CONSTRAINTS["table_x_min"],
        ACTION_CONSTRAINTS["max_lift_radius"],
    )
    y_min = utils.TABLE_CONSTRAINTS["table_y_min"]
    y_max = utils.TABLE_CONSTRAINTS["table_y_max"]
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

    def __init__(self, env: envs.Env, idx_policy: int, arg_objects: List[Object]):
        super().__init__(env=env, idx_policy=idx_policy)
        self._arg_objects = arg_objects

    @property
    def arg_objects(self) -> List[Object]:
        return self._arg_objects

    @abc.abstractmethod
    def execute(self, action: np.ndarray) -> ExecutionResult:
        """Executes the primitive.

        Args:
            action: Normalized action (inside action_space, not action_scale).
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

    def get_policy_args(self) -> Optional[Dict[str, List[int]]]:
        """Gets auxiliary policy args for the current primitive.
        Computes the ordered object indices for the given policy.

        The first index is the end-effector, the following indices are the
        primitive arguments (in order), and the remaining indices are for the
        rest of the objects.

        The non-arg objects can be shuffled randomly for training. This method
        also returns the start and end indices of the non-arg objects.

        Returns:
            Dict with `observation_indices` and `shuffle_range` keys.
        """
        from temporal_policies.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        # Add end-effector index first.
        observation_indices = [TableEnv.EE_OBSERVATION_IDX]

        # Get an ordered list of all other indices besides the end-effector.
        object_to_observation_indices = [
            i
            for i in range(TableEnv.MAX_NUM_OBJECTS)
            if i != TableEnv.EE_OBSERVATION_IDX
        ]
        object_indices = {
            obj: object_to_observation_indices[idx_object]
            for idx_object, obj in enumerate(self.env.real_objects())
        }

        # Add primitive args next.
        observation_indices += [object_indices[obj] for obj in self.arg_objects]
        idx_shuffle_start = len(observation_indices)

        # Add non-null objects next.
        observation_indices += [
            idx_object
            for obj, idx_object in object_indices.items()
            if obj not in self.arg_objects
        ]
        idx_shuffle_end = len(observation_indices)

        # Add all remaining indices in sequential order.
        other_indices: List[Optional[int]] = list(range(TableEnv.MAX_NUM_OBJECTS))
        for i in observation_indices:
            other_indices[i] = None
        observation_indices += [i for i in other_indices if i is not None]

        return {
            "observation_indices": observation_indices,
            "shuffle_range": [idx_shuffle_start, idx_shuffle_end],
        }
        # return self.env.get_policy_args(self)

    def get_non_arg_objects(self, objects: Dict[str, Object]) -> List[Object]:
        """Gets the non-primitive argument objects."""
        return [
            obj
            for obj in objects.values()
            if obj not in self.arg_objects and not obj.isinstance(Null)
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
    def from_action_call(action_call: str, env: envs.Env) -> "Primitive":
        from temporal_policies.envs.pybullet.table_env import TableEnv

        assert isinstance(env, TableEnv)

        name, arg_names = symbolic.parse_proposition(action_call)
        arg_objects = [env.objects[obj_name] for obj_name in arg_names]

        primitive_class = globals()[name.capitalize()]
        idx_policy = env.primitives.index(name)
        return primitive_class(env=env, idx_policy=idx_policy, arg_objects=arg_objects)

    def __eq__(self, other) -> bool:
        if isinstance(other, Primitive):
            return str(self) == str(other)
        else:
            return False

    def __str__(self) -> str:
        args = "" if self.arg_objects is None else ", ".join(map(str, self.arg_objects))
        return f"{type(self).__name__}({args})"


class Pick(Primitive):
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_scale = gym.spaces.Box(*primitive_actions.PickAction.range())
    Action = primitive_actions.PickAction
    ALLOW_COLLISIONS = False

    def execute(self, action: np.ndarray) -> ExecutionResult:
        from temporal_policies.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        # Parse action.
        a = primitive_actions.PickAction(self.scale_action(action))
        dbprint(a)

        # Get object pose.
        obj = self.arg_objects[0]
        obj_pose = obj.pose()
        obj_quat = eigen.Quaterniond(obj_pose.quat)

        # Compute position.
        command_pos = obj_pose.pos + obj_quat * a.pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(a.theta.item(), obj_quat)

        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        objects = self.env.objects
        robot = self.env.robot
        if not self.ALLOW_COLLISIONS:
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
        obj = self.arg_objects[0]
        if obj.isinstance(Hook):
            hook: Hook = obj  # type: ignore
            pos_handle, pos_head, _ = Hook.compute_link_positions(
                hook.head_length, hook.handle_length, hook.handle_y, hook.radius
            )
            action_range = self.Action.range()
            if random.random() < hook.handle_length / (
                hook.handle_length + hook.head_length
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

    def execute(self, action: np.ndarray) -> ExecutionResult:
        from temporal_policies.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        MAX_DROP_DISTANCE = 0.05

        # Parse action.
        a = primitive_actions.PlaceAction(self.scale_action(action))
        dbprint(a)

        obj, target = self.arg_objects

        # Get target pose.
        target_pose = target.pose()
        target_quat = eigen.Quaterniond(target_pose.quat)

        # Scale action to target bbox.
        xy_action_range = primitive_actions.PlaceAction.range()[:, :2]
        xy_normalized = (a.pos[:2] - xy_action_range[0]) / (
            xy_action_range[1] - xy_action_range[0]
        )
        xy_target_range = np.array(target.bbox[:, :2])
        print(xy_target_range)
        if target.name == "table":
            xy_target_range[0, 0] = utils.TABLE_CONSTRAINTS["table_x_min"]
            xy_target_range[1, 0] = ACTION_CONSTRAINTS["max_lift_radius"]
        xy_target = (
            xy_target_range[1] - xy_target_range[0]
        ) * xy_normalized + xy_target_range[0]
        pos = np.append(xy_target, a.pos[2])

        # Compute position.
        command_pos = target_pose.pos + target_quat * pos

        # Compute orientation.
        command_quat = compute_top_down_orientation(a.theta.item(), target_quat)

        pre_pos = np.append(command_pos[:2], ACTION_CONSTRAINTS["max_lift_height"])

        objects = self.env.objects
        robot = self.env.robot
        if not self.ALLOW_COLLISIONS:
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

        self.env.wait_until_stable()

        if utils.is_below_table(obj):
            # Falling off the table is an exception.
            return ExecutionResult(success=False, truncated=True)

        if not utils.is_upright(obj) or not utils.is_above(obj, target):
            return ExecutionResult(success=False, truncated=False)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()
        action_range = action.range()

        # Compute an appropriate place height given the grasped object's height.
        obj = self.arg_objects[0]
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

    def execute(self, action: np.ndarray) -> ExecutionResult:
        from temporal_policies.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        PULL_HEIGHT = 0.03
        MIN_PULL_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PullAction(self.scale_action(action))
        dbprint(a)

        # Get target pose in polar coordinates
        target = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
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
        T_gripper_to_world = self.env.robot.arm.ee_pose().to_eigen()
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

        objects = self.env.objects
        robot = self.env.robot
        if not self.ALLOW_COLLISIONS:
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
                    obj.body_id
                    for obj in self.get_non_arg_objects(objects)
                    if obj.name != "table"
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
            if self.ALLOW_COLLISIONS:
                # No objects should move after lifting the hook.
                did_non_args_move = self.create_non_arg_movement_check(objects)

            robot.goto_pose(post_pos, command_pose_pull.quat)
            if did_non_args_move():
                raise ControlException(
                    f"Robot.goto_pose({post_pos}, {command_pose_pull.quat}) collided"
                )
        except ControlException as e:
            dbprint("Pull.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        self.env.wait_until_stable()

        if not utils.is_upright(target):
            return ExecutionResult(success=False, truncated=False)

        new_target_distance = np.linalg.norm(target.pose().pos[:2])
        if (
            new_target_distance >= target_distance - MIN_PULL_DISTANCE
            or not utils.is_inworkspace(obj=target)
        ):
            return ExecutionResult(success=False, truncated=False)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()

        obj = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
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

    def execute(self, action: np.ndarray) -> ExecutionResult:
        from temporal_policies.envs.pybullet.table_env import TableEnv

        assert isinstance(self.env, TableEnv)

        PUSH_HEIGHT = 0.03
        MIN_PUSH_DISTANCE = 0.01

        # Parse action.
        a = primitive_actions.PushAction(self.scale_action(action))
        dbprint(a)

        # Get target pose in polar coordinates
        target = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
        target_pose = target.pose()
        if target_pose.pos[0] < 0:
            return ExecutionResult(success=False, truncated=True)

        target_distance = np.linalg.norm(target_pose.pos[:2])
        reach_xy = target_pose.pos[:2] / target_distance
        reach_theta = np.arctan2(reach_xy[1], reach_xy[0])
        reach_aa = eigen.AngleAxisd(reach_theta, np.array([0.0, 0.0, 1.0]))
        reach_quat = eigen.Quaterniond(reach_aa)

        robot = self.env.robot
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

        objects = self.env.objects
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
            if self.ALLOW_COLLISIONS:
                # Avoid pushing off the rack.
                did_rack_move = self.create_non_arg_movement_check(
                    {obj.name: obj for obj in objects.values() if obj.isinstance(Rack)}
                )

            robot.goto_pose(
                command_pose_push.pos,
                command_pose_push.quat,
                pos_gains=np.array([[49, 14], [49, 14], [121, 22]]),
            )
            if (not self.ALLOW_COLLISIONS and did_non_args_move()) or (
                self.ALLOW_COLLISIONS and did_rack_move()
            ):
                raise ControlException(
                    f"Robot.goto_pose({command_pose_push.pos}, {command_pose_push.quat}) collided"
                )

            # Target must be pushed a minimum distance.
            new_target_distance = np.linalg.norm(target.pose().pos[:2])
            if new_target_distance <= target_distance + MIN_PUSH_DISTANCE:
                return ExecutionResult(success=False, truncated=True)

            # Target must be pushed underneath rack if it exists.
            if len(self.arg_objects) == 3:
                obj = self.arg_objects[2]
                if obj.isinstance(Rack) and not utils.is_under(target, obj):
                    return ExecutionResult(success=False, truncated=True)

            robot.goto_pose(command_pose_reach.pos, command_pose_reach.quat)

            # Target must be upright.
            if not utils.is_upright(target):
                return ExecutionResult(success=False, truncated=True)

            robot.goto_pose(pre_pos, command_pose_reach.quat)
        except ControlException as e:
            dbprint("Push.execute():\n", e)
            return ExecutionResult(success=False, truncated=True)

        return ExecutionResult(success=True, truncated=False)

    def sample_action(self) -> primitive_actions.PrimitiveAction:
        action = self.Action.random()

        obj = self.arg_objects[0]
        hook: Hook = self.arg_objects[1]  # type: ignore
        obj_halfsize = 0.5 * np.linalg.norm(obj.size[:2])
        collision_length = -0.5 * hook.size[0] - 2 * hook.radius - obj_halfsize
        action.r_reach = collision_length
        action.theta = 0.125 * np.pi

        return action
