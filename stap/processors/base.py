"""
Processors are designed as ways of manipulating entire batches of tensors at once to prepare them for the network.
Examples are as follows:
1. Normalization
2. Image Augmentations applied on the entire batch at once.
"""
import stap


class Processor:
    """
    This is the base processor class. All processors should inherit from it.
    """

    def __init__(self, observation_space):
        # TODO: Remove unused arguments.
        self.training = True
        self.observation_space = observation_space

    def __call__(self, batch):
        # Perform operations on the values. This may be normalization etc.
        raise NotImplementedError

    def unprocess(self, batch):
        raise NotImplementedError

    def to(self, device):
        pass

    @property
    def supports_gpu(self):
        return True

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("Training mode is expected to be a boolean")
        self.training = mode

    def eval(self):
        self.train(mode=False)


class IdentityProcessor(Processor):
    """
    This processor just performs the identity operation
    """

    def __call__(self, batch):
        return batch

    def unprocess(self, batch):
        return batch


class ComposeProcessor(Processor):
    """
    This Processor Composes multiple processors
    """

    def __init__(self, observation_space, processors=[("IdentityProcessor"), {}]):
        super().__init__(observation_space)
        self.processors = []
        for processor_class, processor_kwargs in processors:
            processor_class = vars(stap.processors)[processor_class]
            processor = processor_class(self.observation_space, **processor_kwargs)
            self.processors.append(processor)

    def __call__(self, batch):
        for processor in self.processors:
            batch = processor(batch)
        return batch
