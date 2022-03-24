from typing import Any, Dict, List, Optional, Type

import torch  # type: ignore

import temporal_policies
from temporal_policies.algs import dynamics


def load_dynamics_model(
    dynamics_config: Dict[str, Any],
    task_config: dynamics.TaskConfig,
    policy_checkpoints: List[str],
    dynamics_checkpoint: Optional[str] = None,
    device: str = "auto",
) -> dynamics.DynamicsModel:
    dynamics_class = vars(dynamics)[dynamics_config["dynamics"]]
    assert issubclass(dynamics_class, dynamics.DynamicsModel)

    dataset_class = _parse_class(dynamics_config, "dataset", temporal_policies.datasets)
    network_class = _parse_class(dynamics_config, "network", temporal_policies.networks)
    optimizer_class = _parse_class(dynamics_config, "optim", torch.optim)
    scheduler_class = _parse_class(dynamics_config, "schedule", torch.optim.lr_scheduler)

    network_kwargs = _parse_kwargs(dynamics_config, "network_kwargs")
    dataset_kwargs = _parse_kwargs(dynamics_config, "dataset_kwargs")
    optimizer_kwargs = _parse_kwargs(dynamics_config, "optimizer_kwargs")
    scheduler_kwargs = _parse_kwargs(dynamics_config, "scheduler_kwargs")
    dynamics_kwargs = _parse_kwargs(dynamics_config, "dynamics_kwargs")

    policies = dynamics.load_policies(task_config, policy_checkpoints)

    model = dynamics_class(
        policies,
        network_class=network_class,
        network_kwargs=network_kwargs,
        dataset_class=dataset_class,
        dataset_kwargs=dataset_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        **dynamics_kwargs,
    )
    if dynamics_checkpoint is not None:
        model.load(dynamics_checkpoint, strict=False)

    return model


def _parse_class(config: Dict[str, Any], key: str, module) -> Optional[Type]:
    try:
        return vars(module)[config[key]]
    except KeyError:
        return None


def _parse_kwargs(config: Dict[str, Any], key: str) -> Dict:
    try:
        kwargs = config[key]
    except KeyError:
        return {}
    return {} if kwargs is None else kwargs
