import functools
from typing import Any, Dict, List, Optional, Sequence, Union
import pathlib

import yaml  # type: ignore

from temporal_policies import agents, dynamics, envs
from temporal_policies.utils import configs


class DynamicsFactory(configs.Factory):
    """Dynamics factory."""

    def __init__(
        self,
        dynamics_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        env_factory: Optional[envs.EnvFactory] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
    ):
        """Creates the dynamics model factory from a config or checkpoint.

        Args:
            dynamics_config: Optional dynamics config path or dict. Must be
                provided if checkpoint is None.
            env_factory: Env factory required only for OracleDynamics.
            checkpoint: Optional dynamics checkpoint path. Must be provided if
                dynamics_config is None.
        """
        if checkpoint is not None:
            ckpt_config = dynamics.load_config(checkpoint)
            if dynamics_config is None:
                dynamics_config = ckpt_config
        if dynamics_config is None:
            raise ValueError("Either config or checkpoint must be specified")

        super().__init__(dynamics_config, "dynamics", dynamics)

        if (
            checkpoint is not None
            and self.config["dynamics"] != ckpt_config["dynamics"]
        ):
            raise ValueError(
                f"Config dynamics [{self.config['dynamics']}] and checkpoint"
                f"dynamics [{ckpt_config['dynamics']}] must be the same"
            )

        if issubclass(self.cls, dynamics.LatentDynamics):
            self.kwargs["checkpoint"] = checkpoint

        # # Make sure env is always up to date.
        self._env_factory: Optional[envs.EnvFactory] = None
        if issubclass(self.cls, dynamics.OracleDynamics):
            if env_factory is None:
                raise ValueError("env_factory must not be None for OracleDynamics")
            self._env_factory = env_factory

        self.add_post_hook(functools.partial(self._add_env_factory_hook, env_factory))

    def _add_env_factory_hook(
        self, env_factory: envs.EnvFactory, dynamics_model: dynamics.Dynamics
    ) -> None:
        if not isinstance(dynamics_model, dynamics.OracleDynamics):
            return

        # Make sure OracleDynamics env is always up to date.
        env_factory.add_post_hook(
            functools.partial(dynamics.OracleDynamics.env.__set__, dynamics_model)  # type: ignore
        )

    def __call__(self, *args, **kwargs) -> dynamics.Dynamics:
        """Creates a Dynamics instance.

        *args and **kwargs are transferred directly to the Dynamics constructor.
        DynamicsFactory automatically handles the env and checkpoint arguments.
        """
        if "env" not in kwargs and self._env_factory is not None:
            kwargs["env"] = self._env_factory.get_instance()

        return super().__call__(*args, **kwargs)


def load(
    policies: Sequence[agents.Agent],
    dynamics_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    dynamics_checkpoint: Optional[str] = None,
    env_factory: Optional[envs.EnvFactory] = None,
    device: str = "auto",
) -> dynamics.Dynamics:
    """Loads the dynamics model from a config or checkpoint.

    Args:
        dynamics_config: Optional dynamics config path or dict. Must be
            provided if checkpoint is None.
        env_factory: Env factory required only for OracleDynamics.
        checkpoint: Optional dynamics checkpoint path. Must be provided if
            dynamics_config is None.

    Returns:
        Dynamics instance.
    """
    dynamics_factory = DynamicsFactory(
        dynamics_config=dynamics_config,
        env_factory=env_factory,
        checkpoint=dynamics_checkpoint,
    )
    return dynamics_factory(policies=policies, device=device)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a dynamics config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Dynamics config dict.
    """
    return configs.load_config(path, "dynamics")


# def load(
#     dynamics_config: Dict[str, Any],
#     task_config: dynamics.TaskConfig,
#     policy_checkpoints: Sequence[str],
#     dynamics_checkpoint: Optional[str] = None,
#     device: str = "auto",
# ) -> dynamics.Dynamics:
#     dynamics_class = vars(dynamics)[dynamics_config["dynamics"]]
#     assert issubclass(dynamics_class, dynamics.Dynamics)
#
#     dataset_class = configs.parse_class(
#         dynamics_config, "dataset", temporal_policies.datasets
#     )
#     network_class = configs.parse_class(
#         dynamics_config, "network", temporal_policies.networks
#     )
#     optimizer_class = configs.parse_class(dynamics_config, "optim", torch.optim)
#     scheduler_class = configs.parse_class(
#         dynamics_config, "schedule", torch.optim.lr_scheduler
#     )
#
#     network_kwargs = configs.parse_kwargs(dynamics_config, "network_kwargs")
#     dataset_kwargs = configs.parse_kwargs(dynamics_config, "dataset_kwargs")
#     optimizer_kwargs = configs.parse_kwargs(dynamics_config, "optimizer_kwargs")
#     scheduler_kwargs = configs.parse_kwargs(dynamics_config, "scheduler_kwargs")
#     dynamics_kwargs = configs.parse_kwargs(dynamics_config, "dynamics_kwargs")
#
#     policies = dynamics.load_policies(task_config, policy_checkpoints)
#
#     model = dynamics_class(
#         policies,
#         network_class=network_class,
#         network_kwargs=network_kwargs,
#         dataset_class=dataset_class,
#         dataset_kwargs=dataset_kwargs,
#         optimizer_class=optimizer_class,
#         optimizer_kwargs=optimizer_kwargs,
#         scheduler_class=scheduler_class,
#         scheduler_kwargs=scheduler_kwargs,
#         **dynamics_kwargs,
#     )
#     if dynamics_checkpoint is not None:
#         model.load(dynamics_checkpoint, strict=False)
#
#     return model


# def load_from_path(
#     dynamics_checkpoint: str, policy_checkpoints: Sequence[str], device: str = "auto"
# ) -> dynamics.Dynamics:
#     checkpoint_dir = pathlib.Path(dynamics_checkpoint).parent
#     with open(checkpoint_dir / "dynamics_config.yaml", "r") as f:
#         dynamics_config = yaml.safe_load(f)
#     with open(checkpoint_dir / "exec_config.yaml", "r") as f:
#         task_config = yaml.safe_load(f)
#
#     return load(
#         dynamics_config, task_config, policy_checkpoints, dynamics_checkpoint, device
#     )


def load_policies(
    task_config,
    checkpoint_paths: Sequence[str],
    device: str = "auto",
) -> List[agents.Agent]:
    """Loads the policy checkpoints with deterministic replay buffers to be used
    to train the dynamics model.

    Args:
        task_config: Ordered list of primitive (policy) configs.
        checkpoint_paths: Ordered list of policy checkpoints.
        device: Torch device.

    Returns:
        Ordered list of policies with loaded replay buffers.
    """
    policies: List[agents.Agent] = []
    for checkpoint_path in checkpoint_paths:
        policy = agents.load_from_path(checkpoint_path, device=device, strict=True)
        policy.eval_mode()
        policy.setup_datasets()
        policies.append(policy)

    return policies


def train_dynamics(
    dynamics_config: Dict[str, Any],
    exec_config: Dict[str, Any],
    policy_checkpoints: List[str],
    path: pathlib.Path,
    device: str = "auto",
) -> dynamics.Dynamics:
    print("train_dynamics: Training dynamics model with config:")
    print(dynamics_config)
    path.mkdir(parents=True, exist_ok=False)
    with open(path / "dynamics_config.yaml", "w") as f:
        yaml.dump(dynamics_config, f)
    with open(path / "exec_config.yaml", "w") as f:
        yaml.dump(exec_config, f)
    task_config = exec_config["task"]

    configs.save_git_hash(path)

    model = load(
        dynamics_config=dynamics_config,
        task_config=task_config,
        policy_checkpoints=policy_checkpoints,
        device=device,
    )
    model.train(str(path), **dynamics_config["train_kwargs"])
    return model
