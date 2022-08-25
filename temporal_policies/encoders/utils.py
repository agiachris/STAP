from typing import Any, Dict, Optional, Union
import pathlib

from temporal_policies import encoders, envs
from temporal_policies.utils import configs


class EncoderFactory(configs.Factory):
    """Encoder factory."""

    def __init__(
        self,
        config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Creates the dynamics model factory from a config or checkpoint.

        Args:
            config: Optional dynamics config path or dict. Must be provided if
                checkpoint is None.
            checkpoint: Optional dynamics checkpoint path. Must be provided if
                config is None.
            env: Encoder env.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_config = load_config(checkpoint)
            if config is None:
                config = ckpt_config
            if env is None:
                ckpt_env_config = envs.load_config(checkpoint)
                env = envs.EnvFactory(ckpt_env_config)()

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")
        if env is None:
            raise ValueError("Either env or checkpoint must be specified")

        super().__init__(config, "encoder", encoders)

        if checkpoint is not None and self.config["encoder"] != ckpt_config["encoder"]:
            raise ValueError(
                f"Config encoder [{self.config['encoder']}] and checkpoint"
                f"encoder [{ckpt_config['encoder']}] must be the same"
            )

        self.kwargs["env"] = env
        self.kwargs["device"] = device


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    env: Optional[envs.Env] = None,
    device: str = "auto",
    **kwargs,
) -> encoders.Encoder:
    """Loads the encoder from a config or checkpoint.

    Args:
        config: Optional encoder config path or dict. Must be provided if
            checkpoint is None.
        checkpoint: Optional encoder checkpoint path. Must be provided if
            config is None.
        env: Encoder env.
        device: Torch device.
        kwargs: Optional encoder constructor kwargs.

    Returns:
        Encoder instance.
    """
    encoder_factory = EncoderFactory(
        config=config,
        checkpoint=checkpoint,
        env=env,
        device=device,
    )
    return encoder_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a encoder config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Encoder config dict.
    """
    return configs.load_config(path, "encoder")
