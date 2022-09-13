from typing import Any, Dict, Tuple, List, Optional, Union
import pathlib

from torch import nn

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
        env_kwargs: Dict[str, Any] = {},
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
            env_kwargs: Kwargs passed to EnvFactory for each policy checkpoint.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_config = load_config(checkpoint)
            if config is None:
                config = ckpt_config
            if model_checkpoint is None:
                model_checkpoint = load_model_checkpoint(checkpoint)

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")

        super().__init__(config, "scod", scod)

        if issubclass(self.cls, scod.WrapperSCOD):
            base_config_path = pathlib.Path(self.kwargs.pop("scod_config"))
            base_config = load_config(base_config_path)
            self.kwargs.update(base_config["scod_kwargs"])
            self.config["model_settings"] = base_config["model_settings"]

            if checkpoint is None:
                checkpoint = base_config_path.parent / "final_scod.pt"
                ckpt_config = load_config(checkpoint)
                model_checkpoint = load_model_checkpoint(checkpoint)

            if base_config["scod"] != ckpt_config["scod"]:
                raise ValueError(
                    f"Base config SCOD [{base_config['scod']}] and checkpoint"
                    f"SCOD [{ckpt_config['scod']}] must be the same"
                )
        try:
            sketch_cls = self.kwargs["sketch_cls"]
        except KeyError:
            sketch_cls = "SinglePassPCA"
        self.kwargs["sketch_cls"] = configs.get_class(sketch_cls, scod)

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
            model = agents.load(checkpoint=model_checkpoint, env_kwargs=env_kwargs)

        if model is None:
            raise ValueError(
                "One of config, model, or model_checkpoint must be specified"
            )

        model = setup_model(model, **self.config["model_settings"])
        self.kwargs["model"] = model
        self.kwargs["device"] = device

        self._model_checkpoint = model_checkpoint

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
            f.writelines([f"{self._model_checkpoint}"])


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    model: Optional[agents.Agent] = None,
    model_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    env_kwargs: Dict[str, Any] = {},
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
        env_kwargs: Kwargs passed to EnvFactory for each policy checkpoint.
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
        env_kwargs=env_kwargs,
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


def load_model_checkpoint(path: Union[str, pathlib.Path]) -> Tuple[pathlib.Path]:
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

    return model_checkpoint


def setup_model(
    model: agents.Agent,
    network_name: str = "critic",
    module_index: Optional[int] = None,
    layer_index: Optional[int] = None,
) -> nn.Module:
    """Exctracts a model from the Agent for SCOD.

    Args:
        model: RLAgent model.
        network_name: Network name to extract.
        module_index: Module index if model is comprised of ModuleLists.
        layer_index: All layers up to and including this index will be frozen.

    Returns:
        network: Extracted (sub) model.
    """
    network: nn.Module = getattr(model, network_name)

    if module_index is not None:
        # Assumption: Model consists of submodules stored in a ModuleList container.
        module_lists = [c for c in network.children() if isinstance(c, nn.ModuleList)]
        if not module_lists or len(module_lists) > 1:
            raise ValueError("Network must only contain a single ModuleList")
        network = module_lists[0][module_index]

    if layer_index is not None:
        # Assumption: Network is or contains layers in a Sequential container.
        if isinstance(network, nn.Sequential):
            module = network
        else:
            sequentials = [
                c for c in network.children() if isinstance(c, nn.Sequential)
            ]
            if not sequentials or len(sequentials) > 1:
                raise ValueError("Network must only contain a single Sequential")
            module = sequentials[0]

        num_layers = 0
        for layer in module.children():
            if num_layers == layer_index:
                break
            params = [x for x in layer.parameters()]
            if not params:
                continue
            for p in params:
                p.requires_grad = False
            num_layers += 1

    return network
