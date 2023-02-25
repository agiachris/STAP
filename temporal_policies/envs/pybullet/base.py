from typing import Dict, Any, Tuple, Optional

import os
import sys

from temporal_policies.envs.base import Env
from temporal_policies.envs.pybullet.utils import RedirectStream

with RedirectStream(sys.stderr):
    import pybullet as p


DEFAULT_OPTIONS = {
    "background_color_red": 0.12,
    "background_color_green": 0.12,
    "background_color_blue": 0.25,
}


def to_str_kwarg(kv: Tuple[str, Any]) -> str:
    return f"--{kv[0]}={kv[1]}"


def connect_pybullet(gui: bool = True, gui_kwargs: Dict[str, Any] = {}) -> int:
    if not gui:
        with RedirectStream():
            physics_id = p.connect(p.DIRECT, options=gui_kwargs["options"])
    elif not os.environ["DISPLAY"]:
        raise p.error
    else:
        with RedirectStream():
            physics_id = p.connect(p.GUI, options=gui_kwargs["options"])

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physics_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            0,
            physicsClientId=physics_id,
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=physics_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=physics_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 
            gui_kwargs.get("shadows", 0),
            physicsClientId=physics_id
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=0.25,
            cameraYaw=90,
            cameraPitch=-48,
            cameraTargetPosition=[0.76, 0.07, 0.37],
            physicsClientId=physics_id,
        )
    p.setTimeStep(1 / 60)
    return physics_id


class PybulletEnv(Env):
    def __init__(self, 
        name: str, 
        gui: bool,
        gui_kwargs: Dict[str, Any],
    ):
        self.name = name

        gui_kwargs["options"] = gui_kwargs.get("options", DEFAULT_OPTIONS)
        for k, v in DEFAULT_OPTIONS.items():
            if k not in gui_kwargs["options"]:
                gui_kwargs["options"][k] = v       
        options = " ".join(map(to_str_kwarg, gui_kwargs["options"].items()))
        gui_kwargs["options"] = options
        
        try:
            self._physics_id = connect_pybullet(gui=gui, gui_kwargs=gui_kwargs)
        except p.error as e:
            print(e)
            print("Unable to connect to pybullet with gui. Connecting without gui...")
            self._physics_id = connect_pybullet(gui=False, gui_kwargs=gui_kwargs)

        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_id)

    @property
    def physics_id(self) -> int:
        return self._physics_id

    def close(self) -> None:
        with RedirectStream():
            try:
                p.disconnect(physicsClientId=self.physics_id)
            except (AttributeError, p.error):
                pass

    def __del__(self) -> None:
        self.close()
