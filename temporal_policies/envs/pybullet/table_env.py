import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

from ctrlutils import eigen
import gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet.base import PybulletEnv
from temporal_policies.envs.pybullet.sim import math, robot
from temporal_policies.envs.pybullet.table import object_state, predicates
from temporal_policies.envs.pybullet.table.primitives import Primitive
from temporal_policies.envs.pybullet.table.objects import Object

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
        self._states: Dict[int, Dict[str, Any]] = {}

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
        self._is_recording = False
        self._recording_id = None
        self._recordings: Dict[Any, List[np.ndarray]] = {}
        self._recording_freq = 50
        self._recording_buffer: List[np.ndarray] = []
        self._recording_text = ""
        self._timestep = 0
        # self._frames: List[np.ndarray] = []

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
                "One of primitive, action_call, or (idx_policy, policy_args) must not be None."
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

        if self._is_recording:
            self._recording_text = (
                "Action: ["
                + " ".join([f"{a:.2f}" for a in primitive.scale_action(action)])
                + "]"
            )

        success = primitive.execute(action, self.robot)
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
            self.step_simulation()
            num_iters += 1

        # print("TableEnv.wait_until_stable: {num_iters}")
        return num_iters

    def render_image(
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

    def record_start(self, recording_id: Optional[Any] = None) -> bool:
        """Starts recording the simulation.

        Args:
            recording_id: Prepends the new recording with the existing recording
                saved under this id.
        """
        if isinstance(recording_id, np.ndarray):
            recording_id = recording_id.item()

        if self._is_recording and self._recording_id == recording_id:
            return False

        if recording_id in self._recordings:
            self._recording_buffer = list(self._recordings[recording_id])
        else:
            self._recording_buffer = []

        self._recording_id = recording_id
        self._is_recording = True

        return True

    def record_stop(self, recording_id: Optional[Any] = None) -> bool:
        """Stops recording the simulation.

        Args:
            recording_id: Saves the recording to this id.
        """
        if isinstance(recording_id, np.ndarray):
            recording_id = recording_id.item()

        if not self._is_recording and self._recording_id == recording_id:
            return False

        self._recordings[recording_id] = self._recording_buffer

        self._recording_id = recording_id
        self._is_recording = False

        return True

    def record_save(self, path: Union[str, pathlib.Path], reset: bool = True) -> bool:
        """Saves all the recordings.

        Args:
            path: Path for the recording.
            reset: Reset the recording after saving.
        """
        path = pathlib.Path(path)

        for recording_id, recording in self._recordings.items():
            if len(recording) == 0:
                continue
            if recording_id is not None:
                path_video = path.parent / f"{path.stem}-{recording_id}{path.suffix}"
            else:
                path_video = path
            imageio.mimsave(path_video, recording)

            if reset:
                recording.clear()

        return True

    def step_simulation(self) -> None:
        p.stepSimulation(physicsClientId=self.physics_id)
        if self._is_recording and self._timestep % self._recording_freq == 0:
            self._recording_buffer.append(self.render_image())
        self._timestep += 1
