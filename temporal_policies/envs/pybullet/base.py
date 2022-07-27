import sys

from temporal_policies.envs.base import Env
from temporal_policies.envs.pybullet.utils import RedirectStream

with RedirectStream(sys.stderr):
    import pybullet as p


class PybulletEnv(Env):
    def __init__(self, name: str, gui: bool = True):
        self.name = name
        options = (
            "--background_color_red=0.12 "
            "--background_color_green=0.12 "
            "--background_color_blue=0.25"
        )
        if gui:
            with RedirectStream():
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
            with RedirectStream():
                self._physics_id = p.connect(p.DIRECT, options=options)

        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_id)

    @property
    def physics_id(self) -> int:
        return self._physics_id

    def close(self) -> None:
        with RedirectStream():
            p.disconnect(physicsClientId=self.physics_id)

    def __del__(self) -> None:
        self.close()
