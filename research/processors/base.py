'''
Processors are designed as ways of manipulating entire batches of tensors at once to prepare them for the network.
Examples are as follows:
1. Normalization
2. Image Augmentations applied on the entire batch at once.
'''

class Processor(object):
    '''
    This is the base processor class. All processors should inherit from it.
    '''

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def __call__(self, batch):
        # Perform operations on the values. This may be normalization etc.
        raise NotImplementedError

    def unprocess(self, batch):
        raise NotImplementedError

class Empty(Processor):
    '''
    This processor just performs the identity operation
    '''
    def __call__(self, batch):
        return batch

    def unprocess(self, batch):
        return batch