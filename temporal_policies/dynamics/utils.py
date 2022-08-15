from typing import Any, Dict, List, Optional, Sequence, Union
import pathlib

from temporal_policies import agents, dynamics, envs
from temporal_policies.utils import configs


class DynamicsFactory(configs.Factory):
    """Dynamics factory."""

    def __init__(
        self,
        config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        policies: Optional[Sequence[agents.Agent]] = None,
        policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        env_kwargs: Dict[str, Any] = {},
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Creates the dynamics model factory from a config or checkpoint.

        Args:
            config: Optional dynamics config path or dict. Must be provided if
                checkpoint is None.
            checkpoint: Optional dynamics checkpoint path. Must be provided if
                config is None.
            policies: Optional list of dynamics policies. Must be provided if
                policy_checkpoints is None.
            policy_checkpoints: Optional list of policy checkpoints. Must be
                provided if policies is None.
            env_kwargs: Kwargs passed to EnvFactory for each policy checkpoint.
            env: Env required only for OracleDynamics and TableEnvDynamics.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_config = load_config(checkpoint)
            if config is None:
                config = ckpt_config
            if policy_checkpoints is None:
                policy_checkpoints = load_policy_checkpoints(checkpoint)

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")

        if policies is None and policy_checkpoints is not None:
            policies = [
                agents.load(checkpoint=policy_checkpoint, env_kwargs=env_kwargs)
                for policy_checkpoint in policy_checkpoints
            ]

        if policies is None:
            raise ValueError(
                "One of config, policies, or policy_checkpoints must be specified"
            )

        super().__init__(config, "dynamics", dynamics)

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

        if issubclass(self.cls, (dynamics.OracleDynamics, dynamics.TableEnvDynamics)):
            if issubclass(self.cls, dynamics.OracleDynamics) and env is None:
                raise ValueError(f"{self.cls} requires env to be specified")
            self.kwargs["env"] = env

        self.kwargs["device"] = device
        self.kwargs["policies"] = policies

        self._policy_checkpoints = policy_checkpoints

    def save_config(self, path: Union[str, pathlib.Path]) -> None:
        """Saves the config to path.

        Args:
            path: Directory where config will be saved.
        """
        super().save_config(path)
        if self._policy_checkpoints is None:
            return

        path = pathlib.Path(path)
        with open(path / "policy_checkpoints.txt", "w") as f:
            f.write("\n".join(map(str, self._policy_checkpoints)))


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    policies: Optional[Sequence[agents.Agent]] = None,
    policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    env_kwargs: Dict[str, Any] = {},
    env: Optional[envs.Env] = None,
    device: str = "auto",
    **kwargs,
) -> dynamics.Dynamics:
    """Loads the dynamics model from a config or checkpoint.

    Args:
        config: Optional dynamics config path or dict. Must be provided if
            checkpoint is None.
        checkpoint: Optional dynamics checkpoint path. Must be provided if
            config is None.
        policies: Optional list of dynamics policies. Must be provided if
            policy_checkpoints is None.
        policy_checkpoints: Optional list of policy checkpoints. Must be
            provided if policies is None.
        env_kwargs: Kwargs passed to EnvFactory for each policy checkpoint.
        env: Env required only for OracleDynamics.
        device: Torch device.
        kwargs: Optional dynamics constructor kwargs.

    Returns:
        Dynamics instance.
    """
    dynamics_factory = DynamicsFactory(
        config=config,
        checkpoint=checkpoint,
        policies=policies,
        policy_checkpoints=policy_checkpoints,
        env_kwargs=env_kwargs,
        env=env,
        device=device,
    )
    return dynamics_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a dynamics config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Dynamics config dict.
    """
    return configs.load_config(path, "dynamics")


def load_policy_checkpoints(path: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    """Loads a dynamics config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Dynamics config dict.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.name == "policy_checkpoints.txt":
        policy_checkpoints_path = path
    else:
        if path.suffix == ".pt":
            path = path.parent

        policy_checkpoints_path = path / "policy_checkpoints.txt"

    with open(policy_checkpoints_path, "r") as f:
        policy_checkpoints = [pathlib.Path(line.rstrip()) for line in f]

    return policy_checkpoints
