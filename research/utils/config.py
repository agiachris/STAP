import os
import yaml
import pprint
import importlib
import copy

class Config(object):

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

        # Env Args
        self.config['env'] = None
        self.config['env_kwargs'] = {}

        # Algorithm Args
        self.config['alg'] = None
        self.config['alg_kwargs'] = {}
        self.config['train_kwargs'] = {}
        self.config['seed'] = None # Does nothing right now.

        # Dataset args
        self.config['dataset'] = None
        self.config['dataset_kwargs'] = {}
        self.config['validation_dataset_kwargs'] = None

        # Dataloader arguments
        self.config['collate_fn'] = None
        self.config['batch_size'] = None

        # Optimizer Args
        self.config['optim'] = None
        self.config['optim_kwargs'] = {}
        self.config['scheduler'] = None

        # General arguments
        self.config['checkpoint'] = None

        # network Args
        self.config['network'] = None
        self.config['network_kwargs'] = {}

    def parse(self):
        self.parsed = True
        self.parse_helper(self.config)

    def parse_helper(self, d):
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1 and v[0] == "import":
                # parse the value to an import
                d[k] = getattr(importlib.import_module(v[1]), v[2])
            elif isinstance(v, dict):
                self.parse_helper(v)
        
    def update(self, d):
        self.config.update(d)
    
    def save(self, path):
        if self.parsed:
            print("[CONFIG ERROR] Attempting to saved parsed config. Must save before parsing to classes. ")
            return
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'r') as f:
            data = yaml.load(f)
        config = cls()
        config.update(data)
        return config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __str__(self):
        return pprint.pformat(self.config, indent=4)

    def copy(self):
        assert not self.parsed, "Cannot copy a parsed config"
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config