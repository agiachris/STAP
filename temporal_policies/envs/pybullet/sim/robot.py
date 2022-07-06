from typing import Any, Dict, Optional, Tuple, Union

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
    pass


class Robot(body.Body):
    def __init__(
        self,
        physics_id: int,
        urdf: str,
        arm_kwargs: Dict[str, Any],
        gripper_kwargs: Dict[str, Any],
    ):
        body_id = p.loadURDF(
            fileName=urdf,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
            | p.URDF_MAINTAIN_LINK_ORDER,  # | p.URDF_MERGE_FIXED_LINKS
            physicsClientId=physics_id,
        )
        super().__init__(physics_id, body_id)

        self._arm = arm.Arm(self.physics_id, self.body_id, **arm_kwargs)
        self._gripper = gripper.Gripper(self.physics_id, self.body_id, **gripper_kwargs)

        self.reset()
        self.home_pose = math.Pose.from_eigen(
            dyn.cartesian_pose(self.arm.ab, offset=self.arm.ee_offset)
        )

    @property
    def arm(self) -> arm.Arm:
        return self._arm

    @property
    def gripper(self) -> gripper.Gripper:
        return self._gripper

    def reset(self) -> bool:
        self.gripper.reset()
        self.clear_load()
        return self.arm.reset()

    def clear_load(self) -> None:
        if self.gripper.inertia is not None:
            self.arm.ab.replace_load(self.gripper.inertia)
        else:
            self.arm.ab.clear_load()

    def set_load(self, inertia: dyn.SpatialInertiad) -> None:
        if self.gripper.inertia is not None:
            inertia = inertia + self.gripper.inertia
        self.arm.ab.replace_load(inertia)

    def goto_home(self) -> bool:
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
        self.arm.set_pose_goal(pos, quat, pos_gains, ori_gains, timeout)

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
        self.gripper.remove_grasp_constraint()
        self.clear_load()
        self.gripper.set_grasp(command, pos_gains, timeout)

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
    ) -> bool:
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
        if not self.gripper.create_grasp_constraint(obj.body_id):
            return False

        # Add object load.
        self.set_load(obj.inertia)

        return True
