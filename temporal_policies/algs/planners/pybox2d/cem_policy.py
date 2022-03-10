import torch
import numpy as np

from .pybox2d_base import Box2DTrajOptim


class CEMPolicyPlanner(Box2DTrajOptim):

    def __init__(self, variance, **kwargs):
        """Cross-Entropy Method with randomly sampled actions. Equivalent to RandomShootingPlanner
        while adapting the sampling distribution of actions over several iterations to maximize returns.
        
        args: 
            variance: initial variance for the sampling distribution
        """
        super(CEMPolicyPlanner, self).__init__(**kwargs)
        self._variance = variance

    def plan(self, env, idx, mode="prod"):
        super().plan(env, idx, mode=mode)
        
        action = None
        return action
