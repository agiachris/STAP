from typing import Any, Dict, Optional, Tuple, Union

from ctrlutils import eigen
import numpy as np
import pybullet as p
import spatialdyn as dyn

from temporal_policies.envs.pybullet.sim import (
    articulated_body,
    arm,
    body,
    gripper,
    math,
)


class ControlException(Exception):
    """An exception raised due to a control fault (e.g. reaching singularity)."""

    pass


class Robot(body.Body):
    """User-facing robot interface."""

    def __init__(
        self,
        physics_id: int,
        urdf: str,
        arm_kwargs: Dict[str, Any],
        gripper_kwargs: Dict[str, Any],
    ):
        """Loads the robot from a urdf file.

        Args:
            physics_id: Pybullet physics client id.
            urdf: Path to urdf.
            arm_kwargs: Arm kwargs from yaml config.
            gripper_kwargs: Gripper kwargs from yaml config.
        """
        body_id = p.loadURDF(
            fileName=urdf,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
            | p.URDF_MAINTAIN_LINK_ORDER,  # | p.URDF_MERGE_FIXED_LINKS
            physicsClientId=physics_id,
        )
        super().__init__(physics_id, body_id)

        self._arm = arm.Arm(self.physics_id, self.body_id, **arm_kwargs)
        T_world_to_ee = dyn.cartesian_pose(self.arm.ab).inverse()
        self._gripper = gripper.Gripper(
            self.physics_id, self.body_id, T_world_to_ee, **gripper_kwargs
        )

        self.reset()
        self.home_pose = math.Pose.from_eigen(
            dyn.cartesian_pose(self.arm.ab, offset=self.arm.ee_offset)
        )

    @property
    def arm(self) -> arm.Arm:
        """Controllable arm."""
        return self._arm

    @property
    def gripper(self) -> gripper.Gripper:
        """Controllable gripper."""
        return self._gripper

    def reset(self) -> bool:
        """Resets the robot by setting the arm to its home configuration and the gripper to the open position.

        This method disables torque control and bypasses simulation.
        """
        self.gripper.reset()
        self.clear_load()
        return self.arm.reset()

    def clear_load(self) -> None:
        """Resets the end-effector load to the gripper inertia."""
        if self.gripper.inertia is not None:
            self.arm.ab.replace_load(self.gripper.inertia)
        else:
            self.arm.ab.clear_load()

    def set_load(self, inertia: dyn.SpatialInertiad) -> None:
        """Sets the end-effector load to the sum of the given inertia and gripper inertia."""
        if self.gripper.inertia is not None:
            inertia = inertia + self.gripper.inertia
        self.arm.ab.replace_load(inertia)

    def goto_home(self) -> bool:
        """Uses opspace control to go to the home position."""
        return self.goto_pose(
            self.home_pose.pos,
            self.home_pose.quat,
            pos_gains=(64, 16),
            ori_gains=(64, 16),
        )

    def goto_pose(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[np.ndarray] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Uses opspace control to go to the desired pose.

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Args:
            pos: Optional position. Maintains current position if None.
            quat: Optional quaternion. Maintains current orientation if None.
            pos_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            ori_gains: (kp, kv) gains or [3 x 2] array of xyz gains.
            timeout: Uses the timeout specified in the yaml arm config if None.
        Returns:
            True if the grasp controller converges to the desired position or
            zero velocity, false if the command times out.
        """
        # Set the pose goal.
        self.arm.set_pose_goal(pos, quat, pos_gains, ori_gains, timeout)

        # Simulate until the pose goal is reached.
        status = self.arm.update_torques()
        while status == articulated_body.ControlStatus.IN_PROGRESS:
            self.gripper.update_torques()
            p.stepSimulation(physicsClientId=self.physics_id)
            status = self.arm.update_torques()
        # print("Robot.goto_pose:", pos, quat, status)

        if status == articulated_body.ControlStatus.ABORTED:
            raise ControlException(f"Robot.goto_pose({pos}, {quat})")

        return status in (
            articulated_body.ControlStatus.POS_CONVERGED,
            articulated_body.ControlStatus.VEL_CONVERGED,
        )

    def grasp(
        self,
        command: float,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Sets the gripper to the desired grasp (0.0 open, 1.0 closed).

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Any existing grasp constraints will be cleared and no new ones will be
        created. Use `Robot.grasp_object()` to create a grasp constraint.

        Args:
            command: Desired grasp (range from 0.0 open to 1.0 closed).
            pos_gains: kp gains (only used for sim).
            timeout: Uses the timeout specified in the yaml gripper config if None.
        Returns:
            True if the grasp controller converges to the desired position or
            zero velocity, false if the command times out.
        """
        # Clear any existing grasp constraints.
        self.gripper.remove_grasp_constraint()
        self.clear_load()

        # Set the new grasp command.
        self.gripper.set_grasp(command, pos_gains, timeout)

        # Simulate until the grasp command finishes.
        status = self.gripper.update_torques()
        while status == articulated_body.ControlStatus.IN_PROGRESS:
            self.arm.update_torques()
            p.stepSimulation(physicsClientId=self.physics_id)
            status = self.gripper.update_torques()
        # print("Robot.grasp:", command, status)

        if status == articulated_body.ControlStatus.ABORTED:
            raise ControlException(f"Robot.grasp({command})")

        return status in (
            articulated_body.ControlStatus.POS_CONVERGED,
            articulated_body.ControlStatus.VEL_CONVERGED,
        )

    def grasp_object(
        self,
        obj: body.Body,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
        realistic: bool = True,
    ) -> bool:
        """Attempts to grasp an object and attaches the object to the gripper via a pose constraint.

        This method blocks until the command finishes or times out. A
        ControlException will be raised if the grasp controller is aborted.

        Args:
            command: Desired grasp (range from 0.0 open to 1.0 closed).
            pos_gains: kp gains (only used for sim).
            timeout: Uses the timeout specified in the yaml gripper config if None.
            realistic: If false, creates a pose constraint regardless of whether
                the object is in a secure grasp.
        Returns:
            True if the object is successfully grasped, false otherwise.
        """
        if realistic:
            self.grasp(1, pos_gains, timeout)

            # Wait for grasped object to settle.
            status = self.gripper.update_torques()
            while (
                status
                in (
                    articulated_body.ControlStatus.VEL_CONVERGED,
                    articulated_body.ControlStatus.IN_PROGRESS,
                )
                and self.gripper._iter_timeout >= 0
                and (obj.twist() > 0.001).any()
            ):
                self.arm.update_torques()
                status = self.gripper.update_torques()
                p.stepSimulation(physicsClientId=self.physics_id)

            # Make sure fingers aren't fully closed.
            if status == articulated_body.ControlStatus.POS_CONVERGED:
                return False

        # Lock the object in place with a grasp constraint.
        if not self.gripper.create_grasp_constraint(obj.body_id, realistic):
            return False

        # Add object load.
        T_obj_to_world = obj.pose().to_eigen()
        T_ee_to_world = dyn.cartesian_pose(self.arm.ab)
        T_obj_to_ee = T_ee_to_world.inverse() * T_obj_to_world
        self.set_load(obj.inertia * T_obj_to_ee)

        return True
