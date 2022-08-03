from typing import Any, Dict, Tuple, Optional, Union
import pathlib

from temporal_policies import agents, scod
from temporal_policies.utils import configs


class SCODFactory(configs.Factory):
    """SCOD factory."""

    def __init__(
        self,
        config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        model: Optional[agents.Agent] = None,
        model_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        model_network: Optional[str] = None,
        device: str = "auto",
    ):
        """Creates the SCOD model factory from a config or checkpoint.

        Args:
            config: Optional SCOD config path or dict. Must be provided if
                checkpoint is None.
            checkpoint: Optional SCOD checkpoint path. Must be provided if
                config is None.
            model: Optional SCOD model. Must be provided if
                model_checkpoint is None.
            model_checkpoint: Optional model checkpoint. Must be
                provided if model is None.
            model_network: Model network name. Must be provided as
                if model and model_checkpoint relate an agents.Agent.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_config = load_config(checkpoint)
            if config is None:
                config = ckpt_config
            if model_checkpoint is None or model_network is None:
                model_checkpoint, model_network = load_model_checkpoint(checkpoint)

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")

        super().__init__(config, "scod", scod)

        if issubclass(self.cls, scod.WrapperSCOD):
            base_config_path = pathlib.Path(self.kwargs.pop("scod_config"))
            base_config = load_config(base_config_path)
            self.kwargs.update(base_config["scod_kwargs"])

            if checkpoint is None:
                checkpoint = base_config_path.parent / "final_scod.pt"
                ckpt_config = load_config(checkpoint)
                model_checkpoint, model_network = load_model_checkpoint(checkpoint)

            if base_config["scod"] != ckpt_config["scod"]:
                raise ValueError(
                    f"Base config SCOD [{base_config['scod']}] and checkpoint"
                    f"SCOD [{ckpt_config['scod']}] must be the same"
                )

        if checkpoint is not None:
            self.kwargs["checkpoint"] = checkpoint
            if self.config["scod"] != ckpt_config["scod"] and not issubclass(
                self.cls, scod.WrapperSCOD
            ):
                raise ValueError(
                    f"Config SCOD [{self.config['scod']}] and checkpoint"
                    f"SCOD [{ckpt_config['scod']}] must be the same"
                )

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")

        if model is None and model_checkpoint is not None:
            model = agents.load(checkpoint=model_checkpoint)

        if model is None:
            raise ValueError(
                "One of config, model, or model_checkpoint must be specified"
            )

        if model_network is None:
            raise ValueError(
                "Model network name must be specified to extract nn.Module for SCOD"
            )

        self.kwargs["model"] = getattr(model, model_network)
        self.kwargs["device"] = device

        self._model_checkpoint = model_checkpoint
        self._model_network = model_network

    def save_config(self, path: Union[str, pathlib.Path]) -> None:
        """Saves the config to path.

        Args:
            path: Directory where config will be saved.
        """
        super().save_config(path)
        if self._model_checkpoint is None:
            return

        path = pathlib.Path(path)
        with open(path / "model_checkpoint.txt", "w") as f:
            f.writelines([f"{self._model_checkpoint}\n", f"{self._model_network}"])


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    model: Optional[agents.Agent] = None,
    model_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    model_network: Optional[str] = None,
    device: str = "auto",
    **kwargs,
) -> scod.SCOD:
    """Loads the SCOD model from a config or checkpoint.

    Args:
        config: Optional SCOD config path or dict. Must be provided if
            checkpoint is None.
        checkpoint: Optional SCOD checkpoint path. Must be provided if
                config is None.
        model: Optional SCOD model. Must be provided if
                model_checkpoint is None.
        model_checkpoint: Optional model checkpoint. Must be
                provided if model is None.
        model_network: Model network name. Must be provided as
                if model and model_checkpoint relate an agents.Agent.
        device: Torch device.
        kwargs: Optional SCOD constructor kwargs.

    Returns:
        SCOD instance.
    """
    scod_factory = SCODFactory(
        config=config,
        checkpoint=checkpoint,
        model=model,
        model_checkpoint=model_checkpoint,
        model_network=model_network,
        device=device,
    )
    return scod_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a dynamics config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        SCOD config dict.
    """
    return configs.load_config(path, "scod")


def load_model_checkpoint(path: Union[str, pathlib.Path]) -> Tuple[pathlib.Path, str]:
    """Loads a SCOD config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        SCOD model checkpoint and model network string name.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.name == "model_checkpoint.txt":
        model_checkpoint_path = path
    else:
        if path.suffix == ".pt":
            path = path.parent

        model_checkpoint_path = path / "model_checkpoint.txt"

    with open(model_checkpoint_path, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
        model_checkpoint = pathlib.Path(lines[0])
        model_network = lines[1]

    return model_checkpoint, model_network
