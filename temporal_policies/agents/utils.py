import copy
import importlib
import os
import pathlib
import pprint
import subprocess
from typing import Any, Dict, Optional, Union

import gym
import torch
import yaml

import temporal_policies
from temporal_policies import agents, envs, encoders, schedulers, scod
from temporal_policies.utils import configs


class AgentFactory(configs.Factory):
    """Agent factory."""

    def __init__(
        self,
        config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        env: Optional[envs.Env] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        actor_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        critic_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        encoder_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        scod_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        policy: Optional[agents.Agent] = None,
        env_kwargs: Dict[str, Any] = {},
        device: str = "auto",
    ):
        """Creates the agent factory from an agent config or checkpoint.

        Args:
            config: Optional agent config path or dict. Must be set if
                checkpoint is None.
            env: Optional env. Either env or checkpoint must be specified.
            checkpoint: Policy checkpoint path. Either env or checkpoint must be
                specified.
            encoder_checkpoint: Encoder checkpoint path.
            actor_checkpoint: Checkpoint to pretrained actor (RLAgent).
            critic_checkpoint: Checkpoint to pretrained critic (RLAgent).
            scod_checkpoint: SCOD checkpoint path.
            policy: Optional loaded policy to replace first non-wrapper agent.
            env_kwargs: Kwargs passed to EnvFactory if env is loaded from
                checkpoint.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_agent_config = load_config(checkpoint)
            if config is None:
                config = ckpt_agent_config
            if env is None:
                ckpt_env_config = envs.load_config(checkpoint)
                env = envs.EnvFactory(ckpt_env_config)(**env_kwargs)

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")
        if env is None:
            raise ValueError("Either env or checkpoint must be specified")

        if config == "{AGENT_CONFIG}":
            # DAF doesn't load config from checkpoint.
            assert policy is not None
            self._policy = policy
            return

        super().__init__(config, "agent", agents)

        if issubclass(self.cls, agents.WrapperAgent):
            # Get policy
            policy = agents.load(
                config=self.kwargs["agent_config"],
                env=env,
                checkpoint=checkpoint,
                encoder_checkpoint=encoder_checkpoint,
                scod_checkpoint=scod_checkpoint,
                policy=policy,
                device=device,
            )
            self.kwargs["policy"] = policy
            del self.kwargs["agent_config"]

            # Optionally get SCOD wrapper
            if issubclass(
                self.cls, (agents.SCODCriticAgent, agents.SCODProbabilisticCriticAgent)
            ):
                if "scod_config" not in self.kwargs:
                    raise ValueError(
                        f'AgentFactory requires agent kwarg "scod_config"'
                        f"to construct WrapperAgent {self.cls.__name__}"
                    )
                self.kwargs["scod_wrapper"] = scod.load(
                    config=self.kwargs["scod_config"],
                    model=policy,
                    checkpoint=scod_checkpoint,
                )
                del self.kwargs["scod_config"]

        elif checkpoint is not None:
            if self.config["agent"] != ckpt_agent_config["agent"]:
                raise ValueError(
                    f"Config agent [{self.config['agent']}] and checkpoint "
                    f"agent [{ckpt_agent_config['agent']}] must be the same"
                )

        if encoder_checkpoint is not None:
            self.kwargs["encoder"] = encoders.load(checkpoint=encoder_checkpoint)

        if issubclass(self.cls, agents.RLAgent):
            self.kwargs["checkpoint"] = checkpoint

            if issubclass(self.cls, agents.SAC):
                if critic_checkpoint is not None:
                    # Update critic config / kwargs.
                    critic_config = load_config(critic_checkpoint)
                    for key in ["critic_class", "critic_kwargs"]:
                        self.kwargs[key] = critic_config["agent_kwargs"][key]
                        self.config["agent_kwargs"][key] = critic_config[
                            "agent_kwargs"
                        ][key]

                    self.kwargs["critic"] = agents.load(
                        checkpoint=critic_checkpoint,
                        env=env,
                        encoder_checkpoint=encoder_checkpoint,
                        device=device,
                    ).critic

                if actor_checkpoint is not None:
                    # Update actor config / kwargs.
                    actor_config = load_config(actor_checkpoint)
                    for key in ["actor_class", "actor_kwargs"]:
                        self.kwargs[key] = actor_config["agent_kwargs"][key]
                        self.config["agent_kwargs"][key] = actor_config["agent_kwargs"][
                            key
                        ]

                    self.kwargs["actor"] = agents.load(
                        checkpoint=actor_checkpoint,
                        env=env,
                        encoder_checkpoint=encoder_checkpoint,
                        device=device,
                    ).actor

        self.kwargs["env"] = env
        self.kwargs["device"] = device

    def __call__(self, *args, **kwargs) -> agents.Agent:
        if hasattr(self, "_policy"):
            return self._policy

        return super().__call__(*args, **kwargs)


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env: Optional[envs.Env] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    encoder_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    scod_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    policy: Optional[agents.Agent] = None,
    env_kwargs: Dict[str, Any] = {},
    device: str = "auto",
    **kwargs,
) -> agents.Agent:
    """Loads the agent from an agent config or checkpoint.

    Args:
        config: Optional agent config path or dict. Must be set if checkpoint is
            None.
        env: Optional env. Either env or checkpoint must be specified.
        checkpoint: Policy checkpoint path. Either env or checkpoint must be
            specified.
        scod_checkpoint: SCOD checkpoint path.
        policy: Optional loaded policy to replace first non-wrapper agent.
        env_kwargs: Kwargs passed to EnvFactory if env is loaded from
            checkpoint.
        device: Torch device.
        kwargs: Optional agent constructor kwargs.

    Returns:
        Agent instance.
    """
    agent_factory = AgentFactory(
        config=config,
        env=env,
        checkpoint=checkpoint,
        encoder_checkpoint=encoder_checkpoint,
        scod_checkpoint=scod_checkpoint,
        policy=policy,
        env_kwargs=env_kwargs,
        device=device,
    )
    return agent_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads an agent config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Agent config dict.
    """
    return configs.load_config(path, "agent")


def get_env(env_class, env_kwargs, wrapper, wrapper_kwargs):
    # Try to get the environment
    try:
        env = env_class(**env_kwargs)
    except KeyError:
        env = gym.make(env_class, **env_kwargs)
    if wrapper is not None:
        env = vars(temporal_policies.envs)[wrapper](env, **wrapper_kwargs)
    return env


def get_env_from_config(config):
    # Try to get the environment
    # assert isinstance(config, Config)
    env_class = configs.parse_class(config, "env", temporal_policies.envs)
    if env_class is None:
        return None
    env_kwargs = configs.parse_kwargs(config, "env_kwargs")
    wrapper = config["wrapper"] if "wrapper" in config else None
    wrapper_kwargs = config["wrapper_kwargs"] if "wrapper_kwargs" in config else {}
    return get_env(env_class, env_kwargs, wrapper, wrapper_kwargs)


def get_model(
    config: Dict[str, Any],
    env: Optional[gym.Env] = None,
    eval_env: Optional[gym.Env] = None,
    device="auto",
):
    if isinstance(config, Config):
        config.parse()  # Parse the config
    alg_class = vars(temporal_policies.agents)[config["alg"]]
    dataset_class = (
        None
        if config["dataset"] is None
        else vars(temporal_policies.datasets)[config["dataset"]]
    )
    network_class = (
        None
        if config["network"] is None
        else vars(temporal_policies.networks)[config["network"]]
    )
    optim_class = (
        None if config["optim"] is None else vars(torch.optim)[config["optim"]]
    )
    processor_class = (
        None
        if config["processor"] is None
        else vars(temporal_policies.processors)[config["processor"]]
    )
    env = get_env_from_config(config) if env is None else env
    eval_env = get_env_from_config(config) if eval_env is None else eval_env

    algo = alg_class(
        env,
        network_class,
        dataset_class,
        network_kwargs=config["network_kwargs"],
        dataset_kwargs=config["dataset_kwargs"],
        validation_dataset_kwargs=config["validation_dataset_kwargs"],
        device=device,
        processor_class=processor_class,
        processor_kwargs=config["processor_kwargs"],
        optim_class=optim_class,
        optim_kwargs=config["optim_kwargs"],
        collate_fn=config["collate_fn"],
        batch_size=config["batch_size"],
        checkpoint=config["checkpoint"],
        eval_env=eval_env,
        **config["alg_kwargs"],
    )

    return algo


def train(config, path, device="auto"):
    # Create the save path and save the config
    print("[research] Training agent with config:")
    print(config)
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=False)
    config.save(path)

    # save the git hash
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = process.communicate()[0].strip()
    with open(path / "git_hash.txt", "wb") as f:
        f.write(git_head_hash)

    model = get_model(config, device=device)
    model.path = path
    assert issubclass(type(model), temporal_policies.agents.Agent)
    schedule = (
        None if config["scheduler"] is None else vars(schedulers)[config["scheduler"]]
    )
    model.train(
        path,
        schedule=schedule,
        schedule_kwargs=config["schedule_kwargs"],
        **config["train_kwargs"],
    )
    model.dataset.save()
    model.eval_dataset.save()
    return model


class Config(object):
    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

        # Env Args
        self.config["env"] = None
        self.config["env_kwargs"] = {}

        # Algorithm Args
        self.config["alg"] = None
        self.config["alg_kwargs"] = {}

        # Dataset args
        self.config["dataset"] = None
        self.config["dataset_kwargs"] = {}
        self.config["validation_dataset_kwargs"] = None

        # Dataloader arguments
        self.config["collate_fn"] = None
        self.config["batch_size"] = None

        # Processor arguments
        self.config["processor"] = None
        self.config["processor_kwargs"] = {}

        # Optimizer Args
        self.config["optim"] = None
        self.config["optim_kwargs"] = {}
        self.config["scheduler"] = None

        # network Args
        self.config["network"] = None
        self.config["network_kwargs"] = {}

        # Schedule args
        self.config["schedule"] = None
        self.config["schedule_kwargs"] = {}

        # General arguments
        self.config["checkpoint"] = None
        self.config["seed"] = None  # Does nothing right now.
        self.config["train_kwargs"] = {}

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
            print(
                "[CONFIG ERROR] Attempting to saved parsed config. Must save before parsing to classes. "
            )
            return
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
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
