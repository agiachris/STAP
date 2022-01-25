# Register environment classes here
from .empty import Empty

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register

# Cleanup extra imports
del register
