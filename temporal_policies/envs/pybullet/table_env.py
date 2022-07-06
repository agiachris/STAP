import abc
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import gym
import numpy as np
import pybullet as p
import symbolic
import yaml

from temporal_policies.envs.pybullet.base import PybulletEnv
from temporal_policies.envs.pybullet.sim import body, math, robot
from temporal_policies.envs.pybullet.sim.robot import ControlException


@dataclasses.dataclass
class Object(body.Body):
    name: str
    urdf: str
    is_static: bool
    initial_state: Optional[str]

    def __init__(
        self,
        physics_id: int,
        name: str,
        urdf: str,
        is_static: bool = False,
        initial_state: Optional[str] = None,
    ):
        body_id = p.loadURDF(
            fileName=urdf,
            useFixedBase=is_static,
            physicsClientId=physics_id,
        )
        super().__init__(physics_id, body_id)

        self.name = name
        self.urdf = urdf
        self.is_static = is_static
        self.initial_state = initial_state

        quat_pybullet_to_world = eigen.Quaterniond(super().pose().quat)
        self._modified_axes = not quat_pybullet_to_world.is_approx(
            eigen.Quaterniond.identity()
        )
        if self._modified_axes:
            self._quat_pybullet_to_world = quat_pybullet_to_world
            self._quat_world_to_pybullet = self._quat_pybullet_to_world.inverse()

    def pose(self) -> math.Pose:
        if not self._modified_axes:
            return super().pose()

        return math.Pose.from_eigen(
            self._quat_world_to_pybullet * super().pose().to_eigen()
        )

    def set_pose(self, pose: math.Pose) -> None:
        if not self._modified_axes:
            return super().set_pose(pose)

        return super().set_pose(
            math.Pose.from_eigen(self._quat_pybullet_to_world * pose.to_eigen())
        )

    def load(self) -> None:
        self.body_id = p.loadURDF(
            fileName=self.urdf,
            useFixedBase=self.is_static,
            physicsClientId=self.physics_id,
        )

    def reset(self, objects: Dict[str, "Object"]) -> None:
        if self.is_static or self.initial_state is None:
            return

        predicate, args = symbolic.parse_proposition(self.initial_state)
        if predicate == "on":
            parent_obj = objects[args[1]]
            xyz_min, xyz_max = parent_obj.aabb()
            xyz = np.zeros(3)
            xyz[:2] = np.random.uniform(0.9 * xyz_min[:2], 0.9 * xyz_max[:2])
            xyz[2] = xyz_max[2] + 0.2
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
            self.set_pose(math.Pose(pos=xyz, quat=eigen.Quaterniond(aa).coeffs))


class Primitive(abc.ABC):
    action_space: gym.spaces.Space

    def __init__(self, args: List[Object]):
        self._args = args

    @property
    def args(self) -> List[Object]:
        return self._args

    @abc.abstractmethod
    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        pass

    @staticmethod
    def from_action_call(action_call: str, objects: Dict[str, Object]) -> "Primitive":
        name, arg_names = symbolic.parse_proposition(action_call)
        args = [objects[obj_name] for obj_name in arg_names]

        primitives = {
            "pick": Pick,
            "place": Place,
        }

        return primitives[name](args)

    def __repr__(self) -> str:
        return f"{type(self).__name__.lower()}({', '.join([arg.name for arg in self.args])})"


class Pick(Primitive):
    action_space = gym.spaces.Box(
        low=np.array([-0.05, -0.05, -0.05, -np.pi], dtype=np.float32),
        high=np.array([0.05, 0.05, 0.05, np.pi], dtype=np.float32),
    )

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        print(action.shape)
        # Parse action.
        pos = action[:3]
        theta = action[3]

        # Get object pose.
        obj = self.args[0]
        obj_pose = obj.pose()

        # Compute position.
        command_pos = obj_pose.pos + pos

        # Compute orientation.
        quat_ee_to_obj = eigen.Quaterniond(robot.home_pose.quat)
        quat_obj_to_world = eigen.AngleAxisd(eigen.Quaterniond(obj_pose.quat))
        command_aa = eigen.AngleAxisd(theta, np.array([0.0, 0.0, 1.0]))
        command_quat = quat_obj_to_world * command_aa * quat_ee_to_obj

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


class Place(Primitive):
    action_space = gym.spaces.Box(
        low=np.array([-0.3, -0.5], dtype=np.float32),
        high=np.array([0.9, 0.5], dtype=np.float32),
    )

    def execute(self, action: np.ndarray, robot: robot.Robot) -> bool:
        # Parse action.
        pos = action[:3]
        theta = action[3]

        # Get target pose.
        target = self.args[1]
        target_pose = target.pose()

        # Compute position.
        command_pos = target_pose.pos + pos

        # Compute orientation.
        quat_ee_to_world = eigen.Quaterniond(robot.home_pose.quat)
        target_aa = eigen.AngleAxisd(eigen.Quaterniond(target_pose.quat))
        command_aa = eigen.AngleAxisd(eigen.Quaterniond(robot.home_pose.quat))
        command_aa.angle = target_aa.angle + theta
        command_quat = eigen.Quaterniond(command_aa) * quat_ee_to_world

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


State = Dict[str, np.ndarray]


class TableEnv(PybulletEnv[State, np.ndarray, np.ndarray]):
    # state_space = gym.spaces.Dict()
    image_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    observation_space = gym.spaces.Box(
        low=np.tile(
            np.array([-0.3, -0.5, -0.1, -np.pi, -np.pi, -np.pi], dtype=np.float32), 2
        ),
        high=np.tile(
            np.array([0.9, 0.5, 0.5, np.pi, np.pi, np.pi], dtype=np.float32), 2
        ),
    )

    def __init__(
        self,
        name: str,
        action_skeleton: List[str],
        robot_config: Union[str, Dict[str, Any]],
        objects: Union[str, List[Dict[str, Any]]],
        gui: bool = True,
    ):
        super().__init__(name=name, gui=gui)

        if isinstance(objects, str):
            with open(objects, "r") as f:
                objects = yaml.safe_load(f)
        assert not isinstance(objects, str)
        if isinstance(robot_config, str):
            with open(robot_config, "r") as f:
                robot_config = yaml.safe_load(f)
        assert not isinstance(robot_config, str)

        self._action_skeleton = action_skeleton

        self._robot = robot.Robot(physics_id=self.physics_id, **robot_config)
        p.stepSimulation(self.physics_id)

        self._objects = {
            obj_kwargs["name"]: Object(physics_id=self.physics_id, **obj_kwargs)
            for obj_kwargs in objects
        }

        self._initial_state_id = p.saveState(physicsClientId=self.physics_id)

        self.set_primitive(self.action_skeleton[0])

    @property
    def action_skeleton(self) -> List[str]:
        return self._action_skeleton

    @property
    def robot(self) -> robot.Robot:
        return self._robot

    @property
    def objects(self) -> Dict[str, Object]:
        return self._objects

    @property
    def primitive(self) -> Primitive:
        return self._primitive

    def set_primitive(self, action_call: str) -> None:
        self._primitive = Primitive.from_action_call(action_call, self.objects)
        self.action_space = self.primitive.action_space

    def state(self) -> State:
        obj_states = {}
        for name, obj in self.objects.items():
            pose = obj.pose()
            aa = eigen.AngleAxisd(eigen.Quaterniond(pose.quat))
            obj_states[name] = np.concatenate(
                [pose.pos, aa.angle * aa.axis], dtype=np.float32
            )

        return obj_states

    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        obj_states = self.state()
        arg_states = [obj_states[arg.name] for arg in self.primitive.args]
        return np.concatenate(arg_states, axis=0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        self.robot.reset()
        p.restoreState(stateId=self._initial_state_id, physicsClientId=self.physics_id)

        for obj in self.objects.values():
            obj.reset(self.objects)

        self.wait_until_stable(1)

        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        success = self.primitive.execute(action, self.robot)
        obs = self.get_observation()

        return obs, float(success), True, {}

    def close(self) -> None:
        p.disconnect(physicsClientId=self.physics_id)

    def wait_until_stable(
        self, min_iters: int = 0, max_iters: int = int(3.0 / math.PYBULLET_TIMESTEP)
    ) -> int:
        def is_any_object_moving() -> bool:
            for obj in self.objects.values():
                if (np.abs(obj.twist()) > 0.001).any():
                    return True
            return False

        num_iters = 0
        while num_iters < max_iters and (
            num_iters < min_iters or is_any_object_moving()
        ):
            self.robot.arm.update_torques()
            self.robot.gripper.update_torques()
            p.stepSimulation(physicsClientId=self.physics_id)
            num_iters += 1

        # print("TableEnv.wait_until_stable: {num_iters}")
        return num_iters
        # print("TableEnv.wait_until_stable: {num_iters}")
        return num_iters
