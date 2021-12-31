'''
Processors are designed as ways of manipulating entire batches of tensors at once to prepare them for the network.
Examples are as follows:
1. Normalization
2. Image Augmentations applied on the entire batch at once.
'''
import research
import torch
from research.utils.utils import fetch_from_dict

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

    @property
    def supports_gpu(self):
        return True

class IdentityProcessor(Processor):
    '''
    This processor just performs the identity operation
    '''
    def __call__(self, batch):
        return batch

    def unprocess(self, batch):
        return batch

class ComposeProcessor(Processor):
    '''
    This Processor Composes multiple processors
    '''
    def __init__(self, observation_space, action_space, processors=[("IdentityProcessor"), {}]):
        super().__init__(observation_space, action_space)
        self.processors = []
        for processor_class, processor_kwargs in processors:
            processor_class = vars(research.processors)[processor_class]
            processor = processor_class(self.observation_space, self.action_space, **processor_kwargs)
            self.processors.append(processor)

    def __call__(self, batch):
        for processor in self.processors:
            batch = processor(batch)
        return batch

class ConcatenateKeyProcessor(Processor):
    '''
    This processor gets items from a nested dictionary structure 
    '''
    def __init__(self, observation_space, action_space, dim=1, keys=[]):
        super().__init__(observation_space, action_space)
        self.dim = dim
        self.keys = sorted(keys)

    def __call__(self, batch):
        if isinstance(batch, list):
            return [self(item) for item in batch]
        elif isinstance(batch, tuple):
            return (self(item) for item in batch)
        elif isinstance(batch, dict):
            items = [fetch_from_dict(batch, key) for key in self.keys]
            if len(items) == 0:
                raise ValueError("Did not fetch any keys from the data")
            elif len(items) == 1:
                return items[0]
            else:
                return torch.cat(items, dim=self.dim)
        else:
            raise ValueError("Does not support passed in datatype" + str(type(batch)))
