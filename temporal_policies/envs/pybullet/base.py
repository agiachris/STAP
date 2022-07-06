from typing import Generic

import pybullet as p

from temporal_policies.envs.base import Env
from temporal_policies.utils.typing import StateType, ActType, ObsType


class PybulletEnv(
    Env[StateType, ActType, ObsType], Generic[StateType, ActType, ObsType]
):
    def __init__(self, name: str, gui: bool = True):
        self.name = name
        options = (
            "--background_color_red=0.25 "
            "--background_color_green=0.25 "
            "--background_color_blue=0.25"
        )
        if gui:
            self._physics_id = p.connect(p.GUI, options=options)
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_id
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self.physics_id,
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.physics_id
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.physics_id
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.physics_id
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=0.25,
                cameraYaw=90,
                cameraPitch=-48,
                cameraTargetPosition=[0.76, 0.07, 0.37],
                physicsClientId=self.physics_id,
            )
        else:
            self._physics_id = p.connect(p.DIRECT, options=options)

        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_id)

    @property
    def physics_id(self) -> int:
        return self._physics_id
