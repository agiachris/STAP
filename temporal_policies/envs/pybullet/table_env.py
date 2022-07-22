import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet.base import PybulletEnv
from temporal_policies.envs.pybullet.sim import math, robot
from temporal_policies.envs.pybullet.table import object_state, predicates
from temporal_policies.envs.pybullet.table.primitives import Primitive
from temporal_policies.envs.pybullet.table.objects import Object
from temporal_policies.utils import recording

import pybullet as p  # Import after envs.pybullet.base to avoid print statement.


@dataclasses.dataclass
class CameraView:
    width: int
    height: int
    view_matrix: np.ndarray
    projection_matrix: np.ndarray


class TableEnv(PybulletEnv):
    state_space = gym.spaces.Box(
        low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
    )
    image_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    observation_space = gym.spaces.Box(
        low=np.tile(object_state.ObjectState.range()[0], 3),
        high=np.tile(object_state.ObjectState.range()[1], 3),
    )

    def __init__(
        self,
        name: str,
        primitives: List[str],
        action_skeleton: List[str],
        initial_state: List[str],
        robot_config: Union[str, Dict[str, Any]],
        objects: Union[str, List[Dict[str, Any]]],
        gui: bool = True,
        recording_freq: int = 10,
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
        self._primitives = primitives
        self._initial_state = [
            predicates.Proposition.create(prop) for prop in initial_state
        ]

        self._robot = robot.Robot(
            physics_id=self.physics_id,
            step_simulation_fn=self.step_simulation,
            **robot_config,
        )
        # self.step_simulation()

        object_list = [
            Object.create(physics_id=self.physics_id, **obj_kwargs)
            for obj_kwargs in objects
        ]
        self._objects = {obj.name: obj for obj in object_list}

        self._initial_state_id = p.saveState(physicsClientId=self.physics_id)
        self._states: Dict[int, Dict[str, Any]] = {}  # Saved states.

        self.set_primitive(action_call=self.action_skeleton[0])

        WIDTH, HEIGHT = 405, 270
        PROJECTION_MATRIX = p.computeProjectionMatrixFOV(
            fov=37.8,
            aspect=1.5,
            nearVal=0.02,
            farVal=100,
        )
        self._camera_views = {
            "front": CameraView(
                width=WIDTH,
                height=HEIGHT,
                view_matrix=p.computeViewMatrix(
                    cameraEyePosition=[2.0, 0.0, 1.0],
                    cameraTargetPosition=[0.0, 0.0, 0.1],
                    cameraUpVector=[0.0, 0.0, 1.0],
                ),
                projection_matrix=PROJECTION_MATRIX,
            ),
            "top": CameraView(
                width=WIDTH,
                height=HEIGHT,
                view_matrix=p.computeViewMatrix(
                    cameraEyePosition=[0.3, 0.0, 1.4],
                    cameraTargetPosition=[0.3, 0.0, 0.0],
                    cameraUpVector=[0.0, 1.0, 0.0],
                ),
                projection_matrix=PROJECTION_MATRIX,
            ),
        }

        self._timelapse = recording.Recorder()
        self._recorder = recording.Recorder()
        self._recording_text = ""

    @property
    def action_skeleton(self) -> List[str]:
        return self._action_skeleton

    @property
    def primitives(self) -> List[str]:
        return self._primitives

    @property
    def initial_state(self) -> List[predicates.Proposition]:
        return self._initial_state

    @property
    def robot(self) -> robot.Robot:
        return self._robot

    @property
    def objects(self) -> Dict[str, Object]:
        return self._objects

    def get_primitive(self) -> envs.Primitive:
        return self._primitive

    def set_primitive(
        self,
        primitive: Optional[envs.Primitive] = None,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> envs.Env:
        if primitive is None:
            primitive = self.get_primitive_info(action_call, idx_policy, policy_args)
        assert isinstance(primitive, Primitive)
        self._primitive = primitive

        return self

    def get_primitive_info(
        self,
        action_call: Optional[str] = None,
        idx_policy: Optional[int] = None,
        policy_args: Optional[Any] = None,
    ) -> envs.Primitive:
        if action_call is not None:
            return Primitive.from_action_call(
                action_call, self.primitives, self.objects
            )
        elif idx_policy is not None and policy_args is not None:
            args = ", ".join(obj.name for obj in policy_args)
            action_call = f"{self.primitives[idx_policy]}({args})"
            return Primitive.from_action_call(
                action_call, self.primitives, self.objects
            )
        else:
            raise ValueError(
                "One of action_call or (idx_policy, policy_args) must not be None."
            )

    def get_state(self) -> np.ndarray:
        state_id = p.saveState(physicsClientId=self.physics_id)
        self._states[state_id] = self.robot.get_state()
        return np.array([state_id])

    def set_state(self, state: np.ndarray) -> bool:
        state_id = state.item()
        self.robot.gripper.remove_grasp_constraint()
        p.restoreState(stateId=state_id, physicsClientId=self.physics_id)
        self.robot.set_state(self._states[state_id])
        return True

    def get_observation(self, image: Optional[bool] = None) -> np.ndarray:
        obj_states = self.object_states()
        arg_states = [
            obj_states[arg.name].vector for arg in self.get_primitive().policy_args
        ]
        arg_states.append(obj_states["gripper"].vector)
        return np.concatenate(arg_states, axis=0)

    def object_states(self) -> Dict[str, object_state.ObjectState]:
        state = {obj.name: obj.state() for name, obj in self.objects.items()}

        gripper_state = object_state.ObjectState()
        ee_pose = self.robot.arm.ee_pose()
        aa_pose = eigen.AngleAxisd(eigen.Quaterniond(ee_pose.quat))
        gripper_state.pos = ee_pose.pos
        gripper_state.aa = aa_pose.axis * aa_pose.angle
        state["gripper"] = gripper_state

        return state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        max_attempts: int = 10,
    ) -> np.ndarray:
        for state_id in self._states:
            p.removeState(state_id, physicsClientId=self.physics_id)
        self._states.clear()
        self.set_primitive(action_call=self.action_skeleton[0])

        while True:
            self.robot.reset()
            p.restoreState(
                stateId=self._initial_state_id, physicsClientId=self.physics_id
            )

            for obj in self.objects.values():
                obj.reset()

            if not all(
                any(prop.sample(self.robot, self.objects) for _ in range(max_attempts))
                for prop in self.initial_state
            ):
                continue

            self.wait_until_stable(1)

            if all(prop.value(self.robot, self.objects) for prop in self.initial_state):
                break

        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        primitive = self.get_primitive()
        assert isinstance(primitive, Primitive)

        if self._recorder.is_recording() or self._timelapse.is_recording():
            self._recording_text = (
                "Action: ["
                + ", ".join([f"{a:.2f}" for a in primitive.scale_action(action)])
                + "]"
            )

        self._timelapse.add_frame(self.render)
        success = primitive.execute(action, self.robot)
        obs = self.get_observation()
        self._timelapse.add_frame(self.render)

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
            self.step_simulation()
            num_iters += 1

        # print("TableEnv.wait_until_stable: {num_iters}")
        return num_iters

    def step_simulation(self) -> None:
        p.stepSimulation(physicsClientId=self.physics_id)
        self._recorder.add_frame(self.render)

    def render(
        self, view: str = "front", resolution: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        camera_view = self._camera_views[view]
        width = camera_view.width if resolution is None else resolution[0]
        height = camera_view.height if resolution is None else resolution[1]
        img_rgba = p.getCameraImage(
            width,
            height,
            viewMatrix=camera_view.view_matrix,
            projectionMatrix=camera_view.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_id,
        )[2]
        img_rgba = np.reshape(img_rgba, (height, width, 4))
        img_rgb = img_rgba[:, :, :3]

        img = Image.fromarray(img_rgb, "RGB")
        draw = ImageDraw.Draw(img)
        FONT = ImageFont.truetype("arial.ttf", 15)
        draw.multiline_text(
            (10, 10), str(self.get_primitive()) + f"\n{self._recording_text}", font=FONT
        )

        return np.array(img)

    def record_start(
        self,
        prepend_id: Optional[Any] = None,
        frequency: Optional[int] = None,
        mode: str = "default",
    ) -> bool:
        """Starts recording the simulation.

        Args:
            prepend_id: Prepends the new recording with the existing recording
                saved under this id.
            frequency: Recording frequency.
            mode: Recording mode. Options:
                - 'default': record at fixed frequency.
                - 'timelapse': record timelapse of environment.
        Returns:
            Whether recording was started.
        """
        if isinstance(prepend_id, np.ndarray):
            prepend_id = prepend_id.item()
        if prepend_id is not None:
            prepend_id = str(prepend_id)

        if mode == "timelapse":
            self._timelapse.start(prepend_id)
        elif mode == "default":
            self._recorder.start(prepend_id, frequency)
        else:
            return False

        return True

    def record_stop(self, save_id: Optional[Any] = None, mode: str = "default") -> bool:
        """Stops recording the simulation.

        Args:
            save_id: Saves the recording to this id.
            mode: Recording mode. Options:
                - 'default': record at fixed frequency.
                - 'timelapse': record timelapse of environment.
        Returns:
            Whether recording was stopped.
        """
        if isinstance(save_id, np.ndarray):
            save_id = save_id.item()
        if save_id is not None:
            save_id = str(save_id)

        if mode == "timelapse":
            return self._timelapse.stop(save_id)
        elif mode == "default":
            return self._recorder.stop(save_id)
        else:
            return False

    def record_save(
        self,
        path: Union[str, pathlib.Path],
        reset: bool = True,
        mode: Optional[str] = None,
    ) -> bool:
        """Saves all the recordings.

        Args:
            path: Path for the recording.
            reset: Reset the recording after saving.
            mode: Recording mode to save. If None, saves all recording modes.
        Returns:
            Whether any recordings were saved.
        """
        is_saved = False
        if mode is None or mode == "timelapse":
            is_saved |= self._timelapse.save(path, reset)
        if mode is None or mode == "default":
            is_saved |= self._recorder.save(path, reset)

        return is_saved
