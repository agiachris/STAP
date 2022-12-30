import dataclasses
import itertools
import multiprocessing
import multiprocessing.synchronize
import pathlib
import random
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from temporal_policies.envs import base as envs
from temporal_policies.envs.pybullet import real
from temporal_policies.envs.pybullet.base import PybulletEnv
from temporal_policies.envs.pybullet.real import object_tracker
from temporal_policies.envs.pybullet.sim import math, robot
from temporal_policies.envs.pybullet.table import object_state, predicates, utils
from temporal_policies.envs.pybullet.table.primitives import (
    Primitive,
    Pick,
    Pull,
    Push,
    initialize_robot_pose,
)
from temporal_policies.envs.pybullet.table.objects import Null, Object, ObjectGroup
from temporal_policies.envs.variant import VariantEnv
from temporal_policies.utils import random as random_utils, recording

import pybullet as p  # Import after envs.pybullet.base to avoid print statement.

dbprint = lambda *args: None  # noqa
# dbprint = print


@dataclasses.dataclass
class CameraView:
    width: int
    height: int
    view_matrix: np.ndarray
    projection_matrix: np.ndarray


@dataclasses.dataclass
class Task:
    action_skeleton: List[Primitive]
    initial_state: List[predicates.Predicate]
    prob: float

    @staticmethod
    def create(
        env: "TableEnv",
        action_skeleton: List[str],
        initial_state: List[str],
        prob: Optional[float] = None,
    ) -> "Task":
        primitives = []
        for action_call in action_skeleton:
            primitive = env.get_primitive_info(action_call=action_call)
            assert isinstance(primitive, Primitive)
            primitives.append(primitive)
        propositions = [predicates.Predicate.create(prop) for prop in initial_state]
        return Task(
            action_skeleton=primitives,
            initial_state=propositions,
            prob=float("nan") if prob is None else prob,
        )


class TaskDistribution:
    def __init__(self, env: "TableEnv", tasks: List[Dict[str, Any]]):
        self._tasks = [Task.create(env, **task) for task in tasks]
        assert all(np.isnan(task.prob) for task in self.tasks) or all(
            not np.isnan(task.prob) for task in self.tasks
        )

        # Normalize probabilities.
        self._probabilities = np.array(
            [1.0 if np.isnan(task.prob) else task.prob for task in self.tasks]
        )
        self._probabilities /= self._probabilities.sum()
        for task, prob in zip(self.tasks, self._probabilities):
            task.prob = prob

    @property
    def tasks(self) -> List[Task]:
        return self._tasks

    def sample(self) -> Task:
        idx_task = np.random.choice(len(self.tasks), p=self._probabilities)
        return self.tasks[idx_task]


class TableEnv(PybulletEnv):
    MAX_NUM_OBJECTS = 8  # Number of rows in the observation matrix.
    EE_OBSERVATION_IDX = 0  # Index of the end-effector in the observation matrix.

    state_space = gym.spaces.Box(
        low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
    )
    image_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    # Vector containing num_policy_args + 1 object states, corresponding to the
    # object states for each of the policy_args and an additional object state
    # for the gripper.
    observation_space = gym.spaces.Box(
        low=np.tile(object_state.ObjectState.range()[0], (MAX_NUM_OBJECTS, 1)),
        high=np.tile(object_state.ObjectState.range()[1], (MAX_NUM_OBJECTS, 1)),
    )

    metadata = {
        "render_modes": [
            "default",
            "front",
            "front_high_res",
            "top",
            "top_high_res",
            "front_right",
            "front_right_high_res",
            "profile",
            "profile_high_res",
        ]
    }

    def __init__(
        self,
        name: str,
        primitives: List[str],
        tasks: List[Dict[str, Any]],
        robot_config: Union[str, Dict[str, Any]],
        objects: Union[str, List[Dict[str, Any]]],
        object_groups: Optional[List[Dict[str, Any]]] = None,
        object_tracker_config: Optional[Union[str, Dict[str, Any]]] = None,
        recording_freq: int = 15,
        gui: bool = True,
        num_processes: int = 1,
        reset_queue_size: int = 100,
        child_process_seed: Optional[int] = None,
        use_curriculum: bool = False,
    ):
        """Constructs the TableEnv.

        Args:
            name: Env name.
            primitives: Ordered list of primitive names.
            tasks: List of dicts containing `initial_state` and `action_skeleton` keys.
            robot_config: Config to construct `pybullet.sim.robot.Robot`.
            objects: List of objects in the scene.
            object_groups: List of object groups to use with `Variant` objects.
            object_tracker_config: Config to construct `pybullet.real.object_tracker.ObjectTracker`.
            recording_freq: Recording frequency.
            gui: Whether to open Pybullet gui.
            num_processes: Number of processes to use. One will be dedicated to
                the main environment, while the rest will find valid
                `env.reset()` initializations.
            reset_queue_size: Number of `env.reset()` initializations to keep in
                the queue. Only used if `num_processes` > 1.
            child_process_seed: Random seed to use for the first child process.
                Helpful for deterministic evaluation. Will not be used for the
                main process!
            use_curriculum: Whether to use a curriculum on the number of objects.
        """
        super().__init__(name=name, gui=gui)

        # Launch external reset process.
        if reset_queue_size <= 0 or num_processes <= 1:
            self._process_pipes: Optional[
                List[multiprocessing.connection.Connection]
            ] = None
            self._seed_queue: Optional[
                multiprocessing.Queue[Tuple[int, Optional[dict]]]
            ] = None
            self._seed_buffer = None
            self._reset_processes = None
        else:
            pipes = [multiprocessing.Pipe() for idx_process in range(num_processes - 1)]
            self._process_pipes = [pipe[0] for pipe in pipes]
            self._seed_queue = multiprocessing.Queue()
            self._seed_buffer = multiprocessing.Semaphore(reset_queue_size)
            self._reset_processes = [
                multiprocessing.Process(
                    target=TableEnv._queue_reset_seeds,
                    daemon=True,
                    kwargs={
                        "process_id": (idx_process, num_processes - 1),
                        "pipe": pipe[1],
                        "seed_queue": self._seed_queue,
                        "seed_buffer": self._seed_buffer,
                        "name": name,
                        "primitives": primitives,
                        "tasks": tasks,
                        "robot_config": robot_config,
                        "objects": objects,
                        "object_groups": object_groups,
                        "object_tracker_config": object_tracker_config,
                        "seed": child_process_seed if idx_process == 0 else None,
                    },
                )
                for idx_process, pipe in enumerate(pipes)
            ]
            for process in self._reset_processes:
                process.start()
        self._process_id: Optional[Tuple[int, int]] = None

        # Load configs.
        object_kwargs: List[Dict[str, Any]] = utils.load_config(objects)
        robot_kwargs: Dict[str, Any] = utils.load_config(robot_config)

        # Set primitive names.
        self._primitives = primitives

        # Create robot.
        self._robot = robot.Robot(
            physics_id=self.physics_id,
            step_simulation_fn=self.step_simulation,
            **robot_kwargs,
        )

        # Create object groups.
        if object_groups is None:
            object_group_list = []
        else:
            object_group_list = [
                ObjectGroup(physics_id=self.physics_id, **group_config)
                for group_config in object_groups
            ]
        self._object_groups = {group.name: group for group in object_group_list}

        # Create objects.
        object_list = [
            Object.create(
                physics_id=self.physics_id,
                object_groups=self.object_groups,
                **utils.load_config(obj_config),
            )
            for obj_config in object_kwargs
        ]
        self._objects = {obj.name: obj for obj in object_list}
        # for group in self.object_groups.values():
        #     group.compute_probabilities(self.objects)

        # Load optional object tracker.
        if object_tracker_config is not None:
            object_tracker_kwargs: Dict[str, Any] = utils.load_config(
                object_tracker_config
            )
            self._object_tracker: Optional[
                object_tracker.ObjectTracker
            ] = object_tracker.ObjectTracker(
                objects=self.objects, **object_tracker_kwargs
            )
        else:
            self._object_tracker = None

        # Create tasks.
        self._tasks = TaskDistribution(self, tasks)
        self._task = self.tasks.tasks[0]
        self.set_primitive(self.action_skeleton[0])

        # Initialize pybullet state cache.
        self._initial_state_id = p.saveState(physicsClientId=self.physics_id)
        self._states: Dict[int, Dict[str, Any]] = {}  # Saved states.

        self._use_curriculum = use_curriculum

        # Initialize rendering.
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
            "front_right": CameraView(
                width=WIDTH,
                height=HEIGHT,
                view_matrix=p.computeViewMatrix(
                    cameraEyePosition=[0.8, 1.2, 0.9],
                    cameraTargetPosition=[0.4, 0.2, 0.25],
                    cameraUpVector=[0.0, 0.0, 1.0],
                ),
                projection_matrix=PROJECTION_MATRIX,
            ),
            "profile": CameraView(
                width=WIDTH,
                height=HEIGHT,
                view_matrix=p.computeViewMatrix(
                    cameraEyePosition=[0.30, 1.60, 0.5],
                    cameraTargetPosition=[0.30, 0.1, 0.28],
                    cameraUpVector=[0.0, 0.0, 1.0],
                ),
                projection_matrix=PROJECTION_MATRIX,
            ),
        }
        self.render_mode = "default"

        self._timelapse = recording.Recorder()
        self._recorder = recording.Recorder(recording_freq)
        self._recording_text = ""

    def close(self) -> None:
        try:
            if self._reset_processes is not None:
                for process in self._reset_processes:
                    process.kill()
                for process in self._reset_processes:
                    process.join()
        except AttributeError:
            pass
        super().close()

    @property
    def tasks(self) -> TaskDistribution:
        return self._tasks

    @property
    def task(self) -> Task:
        return self._task

    @property
    def action_skeleton(self) -> Sequence[envs.Primitive]:
        return self.task.action_skeleton

    @property
    def primitives(self) -> List[str]:
        return self._primitives

    @property
    def robot(self) -> robot.Robot:
        return self._robot

    @property
    def objects(self) -> Dict[str, Object]:
        return self._objects

    def real_objects(self) -> Generator[Object, None, None]:
        """Returns an iterator over the non-null objects."""
        return (obj for obj in self.objects.values() if not obj.isinstance(Null))

    @property
    def object_groups(self) -> Dict[str, ObjectGroup]:
        return self._object_groups

    @property
    def object_tracker(self) -> Optional[object_tracker.ObjectTracker]:
        return self._object_tracker

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
            return Primitive.from_action_call(action_call, self)
        elif idx_policy is not None and policy_args is not None:
            arg_indices = [
                idx_obs - 1 if idx_obs > TableEnv.EE_OBSERVATION_IDX else idx_obs
                for idx_obs in policy_args["observation_indices"][
                    : policy_args["shuffle_range"][0]
                ]
                if idx_obs != TableEnv.EE_OBSERVATION_IDX
            ]
            object_names = list(self.objects.keys())
            args = ", ".join(object_names[idx_obj] for idx_obj in arg_indices)
            action_call = f"{self.primitives[idx_policy]}({args})"
            return Primitive.from_action_call(action_call, self)
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
        """Gets the current low-dimensional state for all the objects.

        The observation is a [MAX_NUM_OBJECTS, d] matrix, where d is the length
        of the low-dimensional object state. The first row corresponds to the
        pose of the end-effector, and the following rows correspond to the
        states of all the objects in order. Any remaining rows are zero.
        """
        if image:
            raise NotImplementedError

        obj_states = self.object_states()
        observation = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        for i, obj_state in enumerate(obj_states.values()):
            observation[i] = obj_state.vector
        return observation

    def set_observation(self, observation: np.ndarray) -> None:
        """Sets the object states from the given low-dimensional state observation.

        See `TableEnv.get_observation()` for a description of the observation.
        """
        ee_state = object_state.ObjectState(observation[TableEnv.EE_OBSERVATION_IDX])
        ee_pose = ee_state.pose()
        try:
            self.robot.goto_pose(pos=ee_pose.pos, quat=ee_pose.quat)
        except robot.ControlException:
            print(f"TableEnv.set_observation(): Failed to reach pose {ee_pose}.")

        for idx_observation, object in zip(
            filter(lambda i: i != TableEnv.EE_OBSERVATION_IDX, range(len(observation))),
            self.real_objects(),
        ):
            obj_state = object_state.ObjectState(observation[idx_observation])
            object.set_state(obj_state)

    def object_states(self) -> Dict[str, object_state.ObjectState]:
        """Returns the object states as an ordered dict indexed by object name.

        The first item in the dict corresponds to the end-effector pose.
        """
        state = {}
        for i, obj in enumerate(self.real_objects()):
            if i == TableEnv.EE_OBSERVATION_IDX:
                ee_state = object_state.ObjectState()
                ee_state.set_pose(self.robot.arm.ee_pose())
                state["TableEnv.robot.arm.ee_pose"] = ee_state

            state[obj.name] = obj.state()

        return state

    @staticmethod
    def _queue_reset_seeds(
        process_id: Tuple[int, int],
        pipe: multiprocessing.connection.Connection,
        seed_queue: multiprocessing.Queue,
        seed_buffer: multiprocessing.synchronize.Semaphore,
        name: str,
        primitives: List[str],
        tasks: List[Dict[str, List[str]]],
        robot_config: Union[str, Dict[str, Any]],
        objects: Union[str, List[Dict[str, Any]]],
        object_groups: Optional[List[Dict[str, Any]]],
        object_tracker_config: Optional[Union[str, Dict[str, Any]]],
        seed: Optional[int],
    ) -> None:
        """Queues successful reset seeds in an external process."""
        env = TableEnv(
            name=name,
            primitives=primitives,
            tasks=tasks,
            robot_config=robot_config,
            objects=objects,
            object_groups=object_groups,
            object_tracker_config=object_tracker_config,
            gui=False,
            num_processes=1,
            reset_queue_size=0,
            child_process_seed=None,
        )
        env._process_id = process_id
        options: Optional[dict] = None
        while True:
            seed_buffer.acquire()
            while pipe.poll():
                message = pipe.recv()
                if "options" in message:
                    options = message["options"]

            _, info = env.reset(seed=seed, options=options)
            seed = info["seed"]
            assert isinstance(seed, int)
            # print("PUT seed:", seed, "process:", process_id)
            seed_queue.put((seed, options))
            seed += 1

    def _seed_generator(
        self, seed: Optional[int]
    ) -> Generator[Tuple[int, Optional[dict]], None, None]:
        """Gets the next seed from the multiprocess queue or an incremented seed."""
        MAX_SIMPLE_INT = 2**30  # Largest simple int in Python.
        if self._seed_queue is None:
            # Child process or single process.
            if seed is None:
                # Make sure seeds don't collide across processes.
                if self._process_id is None:
                    idx_process, num_processes = 0, 0
                else:
                    idx_process, num_processes = self._process_id
                seed = random.randint(
                    0, MAX_SIMPLE_INT // (num_processes + 1) * (idx_process + 1)
                )

            # Increment seeds until one results in a valid env initialization.
            for seed in itertools.count(start=seed):
                yield seed, None
        else:
            # Get a successful reset seed from the multiprocess queue.
            while True:
                seed, options = self._seed_queue.get()
                # print("GET seed:", seed, "queue size:", self._seed_queue.qsize())
                assert self._seed_buffer is not None
                self._seed_buffer.release()
                yield seed, options

    def reset(  # type: ignore
        self, *, seed: Optional[int] = None, options: Optional[dict] = None, max_samples_per_trial: int = 100
    ) -> Tuple[np.ndarray, dict]:
        # Parse reset options.
        try:
            max_num_objects: Optional[int] = options["max_num_objects"]  # type: ignore
        except (TypeError, KeyError):
            max_num_objects = None
        if self._use_curriculum and options is not None and "schedule" in options:
            collect_step: int = options["schedule"]
            max_num_objects = collect_step // 10000 + 1
            if self._process_pipes is not None:
                for pipe in self._process_pipes:
                    pipe.send({"options": {"max_num_objects": max_num_objects}})

        # Clear state cache.
        for state_id in self._states:
            p.removeState(state_id, physicsClientId=self.physics_id)
        self._states.clear()

        task_sampling_trials = 0
        for seed, options in self._seed_generator(seed):
            task_sampling_trials += 1
            if options is not None:
                try:
                    max_num_objects: Optional[int] = options["max_num_objects"]  # type: ignore
                except (TypeError, KeyError):
                    max_num_objects = None

            random_utils.seed(seed)

            if task_sampling_trials % max_samples_per_trial == 0:
                if task_sampling_trials > 0:
                    dbprint(f"TableEnv.reset(seed={seed}): \
                        for {self._task}: failed to sample task \
                        propositions after {max_samples_per_trial} trials.")
                self._task = self.tasks.sample()
                self.set_primitive(self.task.action_skeleton[0])

            self.robot.reset()
            p.restoreState(
                stateId=self._initial_state_id, physicsClientId=self.physics_id
            )

            if self.object_tracker is not None and isinstance(
                self.robot.arm, real.arm.Arm
            ):
                # Track objects from the real world.
                self.object_tracker.update_poses()
                break

            # Reset variants and freeze objects so they don't get simulated.
            for object_group in self.object_groups.values():
                num_objects = object_group.reset(
                    self.objects,
                    self.task.action_skeleton,
                    max_num_objects=max_num_objects,
                )
                if max_num_objects is not None:
                    max_num_objects -= num_objects
            for obj in self.objects.values():
                obj.reset(self.task.action_skeleton)
                obj.freeze()

            # Make sure none of the action skeleton args is Null.
            assert not any(
                any(obj.isinstance(Null) for obj in primitive.arg_objects)
                for primitive in self.task.action_skeleton
            )

            # Sample initial state.
            if not all(
                prop.sample(self.robot, self.objects, self.task.initial_state)
                for prop in self.task.initial_state
            ):
                # Continue if a proposition failed after max_attempts.
                dbprint(f"TableEnv.reset(seed={seed}): Failed to sample propositions")
                continue

            # Sample random robot pose.
            for obj in self.real_objects():
                obj.unfreeze()
            if not initialize_robot_pose(self.robot):
                dbprint(f"TableEnv.reset(seed={seed}): Failed to initialize robot")
                continue

            # Check state again after objects have settled.
            num_iters = self.wait_until_stable(
                min_iters=1, max_iters=math.PYBULLET_STEPS_PER_SEC
            )
            if num_iters == math.PYBULLET_STEPS_PER_SEC:
                # Skip if settling takes longer than 1s.
                dbprint(f"TableEnv.reset(seed={seed}): Failed to stabilize")
                continue

            if (
                self._is_any_object_below_table()
                or self._is_any_object_touching_base()
                or self._is_any_object_falling_off_parent()
            ):
                dbprint(f"TableEnv.reset(seed={seed}): Object fell")
                continue

            if all(
                prop.value(self.robot, self.objects, self.task.initial_state)
                for prop in self.task.initial_state
            ):
                # Break if all propositions in the initial state are true.
                break

            dbprint(f"TableEnv.reset(seed={seed}): Failed to satisfy propositions")

        info = {
            "seed": seed,
            "policy_args": self.get_primitive().get_policy_args(),
        }
        self._seed = seed

        return self.get_observation(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        primitive = self.get_primitive()
        assert isinstance(primitive, Primitive)

        if self._recorder.is_recording() or self._timelapse.is_recording():
            self._recording_text = (
                "Action: ["
                + ", ".join([f"{a:.2f}" for a in primitive.scale_action(action)])
                + "]"
            )

        if isinstance(self.robot.arm, real.arm.Arm):
            input("Continue?")

        self._recorder.add_frame(self.render, override_frequency=True)
        self._timelapse.add_frame(self.render)
        result = primitive.execute(
            action, real_world=isinstance(self.robot.arm, real.arm.Arm)
        )

        if (
            self.object_tracker is not None
            and isinstance(self.robot.arm, real.arm.Arm)
            and not isinstance(primitive, Pick)
        ):
            if isinstance(primitive, (Pull, Push)):
                self.object_tracker.update_poses(exclude=[primitive.arg_objects[1]])
            else:
                # Track objects from the real world.
                self.object_tracker.update_poses()

        obs = self.get_observation()
        self._recorder.add_frame(self.render, override_frequency=True)
        self._timelapse.add_frame(self.render)

        reward = float(result.success)
        terminated = not result.truncated
        info = {"policy_args": primitive.get_policy_args()}

        return obs, reward, terminated, result.truncated, info

    def _is_any_object_below_table(self) -> bool:
        return any(
            not obj.is_static and utils.is_below_table(obj)
            for obj in self.real_objects()
        )

    def _is_any_object_falling_off_parent(self) -> bool:
        def is_falling_off(child: Object, parent: Object) -> bool:
            return (
                # Assume on(child, table) has already been checked.
                parent.name != "table"
                and not child.isinstance(Null)
                and not parent.isinstance(Null)
                and not utils.is_above(child, parent)
            )

        return any(
            is_falling_off(*prop.get_arg_objects(self.objects))
            for prop in self.task.initial_state
            if isinstance(prop, predicates.On)
        )

    def _is_any_object_touching_base(self) -> bool:
        return any(
            not obj.is_static and utils.is_touching(self.robot, obj, link_id_a=-1)
            for obj in self.real_objects()
        )

    def wait_until_stable(
        self, min_iters: int = 0, max_iters: int = 3 * math.PYBULLET_STEPS_PER_SEC
    ) -> int:
        IS_MOVING_KEY = "TableEnv.wait_until_stable"

        def is_any_object_moving() -> bool:
            return any(
                not obj.is_static and utils.is_moving(obj, use_history=IS_MOVING_KEY)
                for obj in self.real_objects()
            )

        # Reset history for `utils.is_moving()`.
        utils.TWIST_HISTORY[IS_MOVING_KEY].clear()

        num_iters = 0
        while (
            num_iters == 0  # Need to step at least once to update collisions.
            or num_iters < max_iters
            and (num_iters < min_iters or is_any_object_moving())
            and not self._is_any_object_below_table()
            and not self._is_any_object_touching_base()
            and not self._is_any_object_falling_off_parent()
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

        if self.object_tracker is not None and not isinstance(
            self.robot.arm, real.arm.Arm
        ):
            # Send objects to RedisGl.
            self.object_tracker.send_poses(self.real_objects())

    def render(self) -> np.ndarray:  # type: ignore
        try:
            camera_view = self._camera_views[self.render_mode.replace("_high_res", "")]
        except KeyError:
            camera_view = self._camera_views["front"]

        if "high_res" in self.render_mode:
            width, height = (1620, 1080)
        else:
            width, height = camera_view.width, camera_view.height
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
        try:
            FONT = ImageFont.truetype("arial.ttf", 15)
            print(
                "Could not find arial.ttf (run `apt install msttcorefonts`?). Using default font."
            )
        except OSError:
            FONT = ImageFont.load_default()
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


class VariantTableEnv(VariantEnv, TableEnv):  # type: ignore
    def __init__(self, variants: Sequence[envs.Env]):
        for env in variants:
            assert isinstance(env, TableEnv)
        super().__init__(variants)

    @property
    def env(self) -> TableEnv:
        env = super().env
        assert isinstance(env, TableEnv)
        return env

    @property
    def tasks(self) -> TaskDistribution:
        return self.env.tasks

    @property
    def task(self) -> Task:
        return self.env.task

    @property
    def robot(self) -> robot.Robot:
        return self.env.robot

    @property
    def objects(self) -> Dict[str, Object]:
        return self.env.objects

    def set_observation(self, observation: np.ndarray) -> None:
        return self.env.set_observation(observation)

    def object_states(self) -> Dict[str, object_state.ObjectState]:
        return self.env.object_states()

    def wait_until_stable(
        self, min_iters: int = 0, max_iters: int = int(3.0 / math.PYBULLET_TIMESTEP)
    ) -> int:
        return self.env.wait_until_stable(min_iters, max_iters)

    def step_simulation(self) -> None:
        return self.env.step_simulation()

# Make a subclass of TableEnv
# update the reset method to keep sampling continuous parameters for the task it just sampled from until success?
# use this Env for the pretraining and also training
class UniformResetsTableEnv(TableEnv):
    # or, add a new kwarg to the env to make it sample until it succeeds? 
    # or until it has sampled too many times?
    def reset():
        raise NotImplementedError