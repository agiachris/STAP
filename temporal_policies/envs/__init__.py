# Register environment classes here
from .empty import Empty
from .pybox2d import *

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register

register(
    id="PlaceRight2D-v0",
    entry_point="temporal_policies.envs.pybox2d.placeright2d:PlaceRight2D"
)

# Cleanup extra imports
del register
