import copy
import functools
import importlib
import os
import pathlib
import pprint
import subprocess
from typing import Any, Dict, Optional, Union

import gym  # type: ignore
import torch  # type: ignore
import yaml  # type: ignore

import temporal_policies
from temporal_policies import agents, envs
from temporal_policies.utils import configs, schedules


class AgentFactory(configs.Factory):
    def __init__(
        self,
        agent_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        env: Optional[envs.Env] = None,
        env_factory: Optional[envs.EnvFactory] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
    ):
        """Creates the agent factory from an agent_config or checkpoint.

        Args:
            agent_config: Optional agent config path or dict. Must be set if
                checkpoint is None.
            env: Optional env. One of env, env_factory, and checkpoint must not
                be None.
            env_factory: Optional env factory. One of env, env_factory, and
                checkpoint must not be None.
            checkpoint: Policy checkpoint path. One of env, env_factory, and
                checkpoint must not be None.
        """
        if checkpoint is not None:
            ckpt_agent_config = agents.load_config(checkpoint)
            ckpt_env_config = envs.load_config(checkpoint)
            if agent_config is None:
                agent_config = ckpt_agent_config
            if env is None and env_factory is None:
                env = envs.EnvFactory(ckpt_env_config)()

        if agent_config is None:
            raise ValueError("Either agent_config or checkpoint must be specified")
        if env is None and env_factory is None:
            raise ValueError("One of env, env_factory, or checkpoint must be specified")

        super().__init__(agent_config, "agent", agents)

        if checkpoint is not None:
            if self.config["agent"] != ckpt_agent_config["agent"]:
                raise ValueError(
                    f"Config agent [{self.config['agent']}] and checkpoint "
                    f"agent [{ckpt_agent_config['agent']}] must be the same"
                )
            if (
                env_factory is not None
                and ckpt_env_config is not None
                and env_factory.config["env"] != ckpt_env_config["env"]
            ):
                raise ValueError(
                    f"Config env [{env_factory.config['env']}] and checkpoint "
                    f"env [{ckpt_env_config['env']}] must be the same"
                )

        if "checkpoint" not in self.kwargs and issubclass(self.cls, agents.RLAgent):
            self.kwargs["checkpoint"] = checkpoint

        if env is None and not issubclass(self.cls, agents.OracleAgent):
            assert env_factory is not None
            env = env_factory.get_instance()
        self.kwargs["env"] = env

        # Make sure env is always up to date.
        self._env_factory: Optional[envs.EnvFactory] = None
        if issubclass(self.cls, agents.OracleAgent):
            if env_factory is None:
                raise ValueError("env_factory must not be None for OracleAgent")
            self._env_factory = env_factory

        self.add_post_hook(functools.partial(self._add_env_factory_hook, env_factory))

    def _add_env_factory_hook(
        self, env_factory: envs.EnvFactory, policy: agents.Agent,
    ) -> None:
        """Makes sure OracleAgent env is always up to date."""
        if not isinstance(policy, agents.OracleAgent):
            return

        env_factory.add_post_hook(
            functools.partial(agents.OracleAgent.env.__set__, policy)  # type: ignore
        )

    def __call__(self, *args, **kwargs) -> agents.Agent:
        """Creates a Dynamics instance.

        *args and **kwargs are transferred directly to the Agent constructor.
        AgentFactory automatically handles the env and checkpoint arguments.
        """
        if "env" not in kwargs and self._env_factory is not None:
            kwargs["env"] = self._env_factory.get_instance()

        return super().__call__(*args, **kwargs)


def load(
    agent_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env_factory: Optional[envs.EnvFactory] = None,
    checkpoint: Optional[str] = None,
    device: str = "auto",
) -> agents.Agent:
    """Loads the agent from an agent_config or checkpoint.

    Args:
        agent_config: Optional agent config path or dict. Must be set if
            checkpoint is None.
        env: Optional env. One of env, env_factory, and checkpoint must not
            be None.
        env_factory: Optional env factory. One of env, env_factory, and
            checkpoint must not be None.
        checkpoint: Policy checkpoint path. One of env, env_factory, and
            checkpoint must not be None.

    Returns:
        Agent instance.
    """
    agent_factory = AgentFactory(
        agent_config=agent_config,
        env_factory=env_factory,
        checkpoint=checkpoint,
    )
    return agent_factory(device=device)


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
        None if config["scheduler"] is None else vars(schedules)[config["scheduler"]]
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


# def load(
#     config: Dict[str, Any],
#     model_path: Union[str, pathlib.Path],
#     env: Optional[gym.Env] = None,
#     eval_env: Optional[gym.Env] = None,
#     device="auto",
#     strict=True,
# ):
#     if isinstance(model_path, str):
#         model_path = pathlib.Path(model_path)
#
#     model = get_model(config, env=env, eval_env=eval_env, device=device)
#     model.load(model_path, strict=strict)
#     model.path = model_path.parent
#     return model
#
#
# def load_from_path(
#     checkpoint_path: Union[str, pathlib.Path],
#     env: Optional[gym.Env] = None,
#     eval_env: Optional[gym.Env] = None,
#     device="auto",
#     strict=True,
# ):
#     if isinstance(checkpoint_path, str):
#         checkpoint_path = pathlib.Path(checkpoint_path)
#
#     config = load_config(checkpoint_path)
#     # checkpoint_path = pathlib.Path(checkpoint_path)
#     # config_path = checkpoint_path.parent / "config.yaml"
#     # config = Config.load(config_path)
#     return load(
#         config,
#         checkpoint_path,
#         env=env,
#         eval_env=eval_env,
#         device=device,
#         strict=strict,
#     )


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
