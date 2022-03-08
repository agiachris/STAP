import torch
import numpy as np

from .pybox2d_base import Box2DTrajOptim


class CEMPolicyPlanner(Box2DTrajOptim):

    def __init__(self, **kwargs):
        super(CEMPolicyPlanner, self).__init__(**kwargs)

    def plan(self, env, idx, samples, mode="prod"):
        """Perform Policy Initialized Cross-Entropy Method. Equivalent to policy shooting and iteratively updating
        and improving the sampling distribution over the specified number of iterations.
        """
        super().plan(env, idx, mode=mode)
        
        action = None
        return action
