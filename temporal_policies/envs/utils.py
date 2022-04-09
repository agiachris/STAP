import pathlib
from typing import Any, Dict, List, Union

from temporal_policies import envs
from temporal_policies.utils import configs


class EnvFactory(configs.Factory):
    """Env factory."""

    def __init__(
        self,
        env_config: Union[str, pathlib.Path, Dict[str, Any]],
    ):
        """Creates the env factory from an env_config.

        Args:
            env_config: Env config path or dict.
        """
        super().__init__(env_config, "env", envs)

        if issubclass(self.cls, envs.pybox2d.Sequential2D):
            self.kwargs["env_factories"] = [
                envs.EnvFactory(env_config) for env_config in self.kwargs["env_configs"]
            ]
            del self.kwargs["env_configs"]

    @property
    def env_factories(self) -> List["EnvFactory"]:
        """Primitive env factories for sequential env."""
        if not issubclass(self.cls, envs.pybox2d.Sequential2D):
            raise AttributeError("Only Sequential2D has attribute env_factories")
        return self.kwargs["env_factories"]


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads an env config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Env config dict.
    """
    return configs.load_config(path, "env")
