import pathlib
from typing import Any, Dict, Union

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


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads an env config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Env config dict.
    """
    return configs.load_config(path, "env")
