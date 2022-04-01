import pathlib
import subprocess

import gym
import torch

import temporal_policies
from . import schedules
from .config import Config

def get_env(env, env_kwargs, wrapper, wrapper_kwargs):
    # Try to get the environment
    try:
        env = vars(temporal_policies.envs)[env](**env_kwargs)
    except:
        env = gym.make(env, **env_kwargs)
    if wrapper is not None:
        env = vars(temporal_policies.envs)[wrapper](env, **wrapper_kwargs)
    return env

def get_env_from_config(config):
    # Try to get the environment
    assert isinstance(config, Config)
    env = config['env']
    env_kwargs = config['env_kwargs']
    wrapper = config['wrapper'] if 'wrapper' in config else None
    wrapper_kwargs = config['wrapper_kwargs'] if 'wrapper_kwargs' in config else {}
    return get_env(env, env_kwargs, wrapper, wrapper_kwargs)

def get_model(config, device="auto"):
    assert isinstance(config, Config)
    config.parse() # Parse the config
    alg_class = vars(temporal_policies.agents)[config['alg']]
    dataset_class = None if config['dataset'] is None else vars(temporal_policies.datasets)[config['dataset']]
    network_class = None if config['network'] is None else vars(temporal_policies.networks)[config['network']]
    optim_class = None if config['optim'] is None else vars(torch.optim)[config['optim']]
    processor_class = None if config['processor'] is None else vars(temporal_policies.processors)[config['processor']]
    env = None if config['env'] is None else get_env_from_config(config)
    eval_env = None if config['env'] is None else get_env_from_config(config)

    algo = alg_class(env, network_class, dataset_class,
                     network_kwargs=config['network_kwargs'], 
                     dataset_kwargs=config['dataset_kwargs'], 
                     validation_dataset_kwargs=config['validation_dataset_kwargs'], 
                     device=device,
                     processor_class=processor_class,
                     processor_kwargs=config['processor_kwargs'],
                     optim_class=optim_class,
                     optim_kwargs=config['optim_kwargs'],
                     collate_fn=config['collate_fn'],
                     batch_size=config['batch_size'],
                     checkpoint=config['checkpoint'],
                     eval_env=eval_env,
                     **config['alg_kwargs'],)
    
    return algo

def train(config, path, device="auto"):
    # Create the save path and save the config
    print("[research] Training agent with config:")
    print(config)
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=False)
    config.save(path)

    # save the git hash
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    with open(path / 'git_hash.txt', 'wb') as f:
        f.write(git_head_hash)
    
    model = get_model(config, device=device)
    model.path = path
    assert issubclass(type(model), temporal_policies.agents.Agent)
    schedule = None if config['scheduler'] is None else vars(schedules)[config['scheduler']]
    model.train(path, schedule=schedule, schedule_kwargs=config['schedule_kwargs'], **config['train_kwargs'])
    model.dataset.save()
    model.eval_dataset.save()
    return model

def load(config, model_path, device="auto", strict=True):
    model = get_model(config, device=device)
    model.load(model_path, strict=strict)
    model.path = model_path.parent
    return model

def load_from_path(checkpoint_path, device="auto", strict=True):
    checkpoint_path = pathlib.Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.yaml"
    config = Config.load(config_path)
    return load(config, checkpoint_path, device=device, strict=strict)
