import inspect

# Register environment classes here
from .empty import Empty
from . import pybox2d
from .pybox2d.base import Box2DBase

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register

for k, v in vars(pybox2d).items():
    if inspect.isclass(v) and issubclass(v, Box2DBase):
        register(
            id="{}-v0".format(k),
            entry_point="temporal_policies.envs.pybox2d.{}:{}".format(k.lower(), k)
        )

# Cleanup extra imports
del register
