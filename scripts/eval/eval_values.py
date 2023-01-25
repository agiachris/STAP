from typing import Any, Dict, Optional, Union, List

import yaml
import pathlib
import argparse
import torch
import numpy as np
from collections import defaultdict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from temporal_policies.utils import configs, random, tensors
from temporal_policies import agents, trainers, datasets
from temporal_policies.networks.encoders import IMAGE_ENCODERS


METRIC_COLORS = {
    "q_freq": "tab:blue",
    "q_mean": "tab:green",
    "q_std": "tab:purple",
    "q_max": "tab:orange",
    "q_min": "tab:red",
}


def barplot(
    path: Union[str, pathlib.Path],
    y_arr: Union[List[float], np.ndarray],
    y_err: Optional[Union[List[float], np.ndarray]] = None,
    x_arr: Optional[Union[List[float], np.ndarray]] = None,
    x_ticks: Optional[List[str]] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    color: Optional[Union[str, List[str]]] = None,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x_arr, y_arr, yerr=y_err, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_arr, x_ticks)
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close()


def evaluate_values(
    path: Union[str, pathlib.Path],
    dataset_config: Union[str, pathlib.Path, Dict[str, Any]],
    dataset_checkpoints: List[Union[str, pathlib.Path]],
    trainer_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    agent_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    num_eval_steps: Optional[int] = None,
    num_bins: Optional[int] = None,
    expected: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
    name: Optional[str] = None,
):
    path = pathlib.Path(path)
    if name is not None:
        path = path / name
    path.mkdir(parents=True, exist_ok=overwrite)
    
    # Load pretrained agent.
    if trainer_checkpoint is not None:
        trainer_factory = trainers.TrainerFactory(checkpoint=trainer_checkpoint, device=device)
        trainer: Union[trainers.ValueTrainer, trainers.PolicyTrainer, trainers.AgentTrainer] = trainer_factory()
        agent = trainer.model
    elif agent_checkpoint is not None:
        if seed is not None:
            random.seed(seed)
        agent_factory = agents.AgentFactory(checkpoint=agent_checkpoint, device=device)
        agent = agent_factory()
    else:
        raise ValueError("Must provide one of trainer checkpoint or agent checkpoint.")
    agent.eval_mode()

    # Load dataset.
    dataset_factory = configs.Factory(dataset_config, "dataset", datasets)
    dataset: datasets.ReplayBuffer = dataset_factory(
        action_space=agent.action_space,
        observation_space=agent.observation_space,
        save_frequency=None,
    )
    for dataset_checkpoint in dataset_checkpoints:
        dataset.load(pathlib.Path(dataset_checkpoint))
    pin_memory = agent.device.type == "cuda"
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        pin_memory=pin_memory,
    )

    metrics: Dict[str, List[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for step, batch in enumerate(iter(dataloader)):
            if num_eval_steps is not None and step > num_eval_steps:
                break
            
            batch = tensors.to(batch, agent.device)
            t_observation = batch["observation"]
            if isinstance(agent.encoder, IMAGE_ENCODERS):
                t_observation = tensors.rgb_to_cnn(t_observation)
            t_state = agent.encoder.encode(t_observation, batch["policy_args"])
            
            # Size [B, critic.num_q_functions]
            batch_qs = torch.stack(agent.critic.forward(t_state, batch["action"])).T
            # Store per-example metrics of size [B,]
            metrics["q_mean"].append(batch_qs.mean(dim=-1).cpu().numpy())
            metrics["q_std"].append(batch_qs.std(dim=-1).cpu().numpy())
            metrics["q_max"].append(torch.max(batch_qs, dim=-1).values.cpu().numpy())
            metrics["q_min"].append(torch.min(batch_qs, dim=-1).values.cpu().numpy())

    # Concatenate batched metrics.
    metrics: Dict[str, np.ndarray] = {k: np.concatenate(v) for k, v in metrics.items()}
    with open(path / "metrics.npz", "wb") as f:
        np.save(f, metrics, allow_pickle=True)

    # Bin metrics based on mean Q-values.
    num_bins = 10 if num_bins is None else num_bins
    bins = np.linspace(0, 1, num_bins + 1)
    binned_metrics: List[Dict[str, np.ndarray]] = []
    bin_metric = "q_mean" if expected else "q_min"
    for i in range(num_bins):        
        mask = np.logical_and(metrics[bin_metric] >= bins[i], metrics[bin_metric] < bins[i+1])
        q_freq = np.sum(mask)
        collect_metric = dict(q_freq=q_freq)
        if q_freq > 0:
            collect_metric.update({k: v[mask] for k, v in metrics.items()})
        else:
            collect_metric.update({k: np.zeros(1) for k in metrics.keys()})
        binned_metrics.append(collect_metric)
    
    # Plot metrics across bins.
    x_arr = np.arange(num_bins)
    x_ticks = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(num_bins)]
    x_label = " ".join(bin_metric.split("_")).title()
    for k in list(metrics.keys()) + ["q_freq"]:
        metric_name = " ".join(k.split("_")).title()
        barplot(
            path=path / f"{k}_ablation_plot.pdf",
            y_arr=[binned_metrics[i][k].mean() for i in range(num_bins)],
            y_err=[binned_metrics[i][k].std() for i in range(num_bins)],
            x_arr=x_arr,
            x_ticks=x_ticks,
            title=f"Q-Ensemble Ablation: {name.capitalize()} {metric_name}",
            x_label=x_label,
            y_label=metric_name,
            color=METRIC_COLORS[k],
        )


def main(args: argparse.Namespace) -> None:
    evaluate_values(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Experiment save path.")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset configuration file.")
    parser.add_argument("--dataset-checkpoints", required=True, nargs="+", help="Paths to data checkpoints.")
    parser.add_argument("--trainer-checkpoint", "-t", help="Path to pretrained agent.")
    parser.add_argument("--agent-checkpoint", "-a", help="Path to agent checkpoint.")
    parser.add_argument("--num-eval-steps", type=int, help="Number of steps to evaluate over.")
    parser.add_argument("--num-bins", type=int, help="Number of bins for Q-value scores.")
    parser.add_argument("--expected", action="store_true", help="Bin Q-ensembles means instead of minimums.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--name", type=str, help="Experiment name")
    main(parser.parse_args())
