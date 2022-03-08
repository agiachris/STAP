import torch
import numpy as np

from .pybox2d_base import Box2DTrajOptim


class CEMRandomPlanner(Box2DTrajOptim):

    def __init__(self, **kwargs):
        super(CEMRandomPlanner, self).__init__(**kwargs)

    def plan(self, env, idx, samples, mode="prod"):
        """Perform the Cross-Entropy Method. Equivalent to random shooting and iteratively updating 
        and improving the sampling distribution over the specified number of iterations.
        """
        super().plan(env, idx, mode=mode)
        
        action = None
        return action
