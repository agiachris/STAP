from typing import Any, Callable, Dict, List, Mapping

import numpy as np
import torch

from temporal_policies.utils import nest, typing


METRIC_CHOICE_FNS = {
    "accuracy": max,
    "reward": max,
    "success": max,
    "loss": min,
    "l2_loss": min,
    "elbo_loss": min,
    "q_loss": min,
    "q1_loss": min,
    "q2_loss": min,
    "actor_loss": min,
    "alpha_loss": min,
    "entropy": min,
    "alpha": min,
    "target_q": max,
    "length": min,
}

METRIC_AGGREGATION_FNS: Dict[str, Callable[[np.ndarray], float]] = {
    "accuracy": np.mean,
    "reward": np.sum,
    "success": lambda x: x[-1],
    "loss": np.mean,
    "l2_loss": np.mean,
    "elbo_loss": np.mean,
    "q_loss": np.mean,
    "q1_loss": np.mean,
    "q2_loss": np.mean,
    "actor_loss": np.mean,
    "alpha_loss": np.mean,
    "entropy": np.mean,
    "alpha": np.mean,
    "target_q": np.mean,
    "length": np.sum,
}


def init_metric(metric: str) -> float:
    """Returns the initial value for the metric.

    Args:
        metric: Metric type.

    Returns:
        inf for min metrics, -inf for max metrics.
    """
    a = -float("inf")
    b = float("inf")
    return b if METRIC_CHOICE_FNS[metric](a, b) == a else a


def best_metric(metric: str, *values) -> float:
    """Returns the best metric value.

    Args:
        metric: Metric type.
        values: Values to compare.

    Returns:
        Min or max value depending on the metric.
    """
    return METRIC_CHOICE_FNS[metric](*values)


def aggregate_metric(metric: str, values: np.ndarray) -> float:
    """Aggregates the metric values.

    Args:
        metric: Metric type.
        values: Values to aggregate.

    Returns:
        Aggregated value.
    """
    return METRIC_AGGREGATION_FNS[metric](values)


def aggregate_metrics(metrics_list: List[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregates a list of metric value dicts.

    Args:
        metric_list: List of metric value dicts.

    Returns:
        Aggregated metric value dict.
    """
    metrics = collect_metrics(metrics_list)
    aggregated_metrics = {
        metric: aggregate_metric(metric, values) for metric, values in metrics.items()
    }
    return aggregated_metrics


def collect_metrics(metrics_list: List[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Transforms a list of metric value dicts to a dict of metric value arrays.

    Args:
        metric_list: List of metric value dicts.

    Returns:
        Dict of metric value arrays.
    """

    def stack(*args):
        args = [arg for arg in args if arg is not None]
        if isinstance(args[0], torch.Tensor):
            return torch.stack(args, dim=0)
        return np.array(args)

    metrics = nest.map_structure(
        stack, *metrics_list, atom_type=(*typing.scalars, np.ndarray, torch.Tensor)
    )
    return metrics
