#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Sequence, Union

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore


def load_results(
    path: Union[str, pathlib.Path], methods: Sequence[str]
) -> Dict[str, List[Dict[str, Any]]]:
    path = pathlib.Path(path)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for method_name in methods:
        results[method_name] = []
        for npz_file in (path / method_name).glob("results_*.npz"):
            with open(npz_file, "rb") as f:
                results[method_name].append(dict(np.load(f, allow_pickle=True)))

    return results


def create_dataframes(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    def get_method_label(method: str) -> str:
        if method == "random":
            return "Random"

        tokens = method.split("_")
        policy = tokens[0]
        planner = tokens[1]
        if planner == "cem":
            planner = planner.upper()

        return f"{policy.capitalize()} {planner}"

    def get_value_label(method: str) -> str:
        if method == "random":
            return "NA"

        tokens = method.split("_")
        if "oracle" in tokens and "value" in tokens:
            return "Oracle"

        return "Q-value"

    def get_dynamics_label(method: str) -> str:
        if method == "random":
            return "NA"

        tokens = method.split("_")
        if "oracle" in tokens and "dynamics" in tokens:
            return "Oracle"

        return "Latent"

    df_plans: Dict[str, List[Any]] = {
        "Method": [],
        "Value": [],
        "Dynamics": [],
        "Predicted success": [],
        "Ground truth success": [],
        "Num sampled": [],
        "Time": [],
        "Position": [],
        "Angle": [],
    }
    for method, method_results in results.items():
        for result in method_results:
            df_plans["Method"].append(get_method_label(method))
            df_plans["Value"].append(get_value_label(method))
            df_plans["Dynamics"].append(get_dynamics_label(method))
            df_plans["Predicted success"].append(result["p_success"].item())
            df_plans["Ground truth success"].append(result["rewards"].prod())
            df_plans["Num sampled"].append(result["p_visited_success"].shape[0])
            df_plans["Time"].append(result["t_planner"].item())
            df_plans["Position"].append(result["scaled_actions"][0, 0])
            df_plans["Angle"].append(result["scaled_actions"][0, 1])

    df_samples: Dict[str, List[Any]] = {
        "Method": [],
        "Value": [],
        "Dynamics": [],
        "Predicted success": [],
        "Position": [],
        "Angle": [],
    }
    for method, method_results in results.items():
        for result in method_results:
            actions = result["scaled_visited_actions"][:, 0]
            num_samples = actions.shape[0]
            df_samples["Method"] += [get_method_label(method)] * num_samples
            df_samples["Value"] += [get_value_label(method)] * num_samples
            df_samples["Dynamics"] += [get_dynamics_label(method)] * num_samples

            # np.ndarray.tolist() converts to float64. Use list comprehension to
            # preserve data type.
            df_samples["Predicted success"] += [x for x in result["p_visited_success"]]
            df_samples["Position"] += [x for x in actions[:, 0]]
            df_samples["Angle"] += [x for x in actions[:, 1]]

    return pd.DataFrame(df_plans), pd.DataFrame(df_samples)


def plot_planning_results(
    df_plans: pd.DataFrame, df_samples: pd.DataFrame, path: Union[str, pathlib.Path]
) -> None:
    def barplot(ax: plt.Axes, df_plans: pd.DataFrame, **kwargs) -> plt.Axes:
        sns.barplot(ax=ax, data=df_plans, **kwargs)
        ax.set_xticklabels(
            [label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()]
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_axisbelow(True)
        ax.grid(axis="y")

        # Change colors and shift bars.
        num_methods = len(df_plans["Method"].unique())
        for idx_bar, (bar, line) in enumerate(zip(ax.patches, ax.lines)):
            idx_method = idx_bar % num_methods
            idx_value_dynamics = idx_bar // num_methods

            # Compute color.
            a = idx_value_dynamics * 0.4 if idx_value_dynamics < 3 else 0
            color = 0.8 * np.array(sns.color_palette()[idx_method])
            color = (1 - a) * color + a * np.ones(3)

            # Compute position.
            dx = 0.1 if idx_value_dynamics < 3 else -0.3

            # Modify plot.
            bar.set_color(color)
            bar.set_x(bar.get_x() + dx)
            line.set_xdata(line.get_xdata() + dx)

    df_plans = df_plans.copy()
    df_plans["Value / Dynamics"] = df_plans.apply(
        lambda x: f"{x['Value']} / {x['Dynamics']}", axis=1
    )
    df_plans["Predicted success error"] = (
        df_plans["Predicted success"] - df_plans["Ground truth success"]
    )

    # Change Random method's value function to oracle.
    idx_random = df_plans["Method"] == "Random"
    df_plans.loc[idx_random, "Predicted success"] = 0.0
    df_plans.loc[idx_random, "Predicted success error"] = 0.0

    df_samples = df_samples.copy()
    df_samples["Value / Dynamics"] = df_samples.apply(
        lambda x: f"{x['Value']} / {x['Dynamics']}", axis=1
    )

    # Change Random method's value function to oracle.
    idx_samples_random = df_samples["Method"] == "Random"
    assert idx_random.sum() == idx_samples_random.sum()
    df_samples.loc[idx_samples_random, "Predicted success"] = df_plans[
        "Ground truth success"
    ][idx_random].to_numpy()

    df_samples["Predicted success > 0.5"] = df_samples["Predicted success"] > 0.5

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    ax = axes[0, 0]
    barplot(ax, df_plans, x="Method", y="Ground truth success", hue="Value / Dynamics")
    ax.set_title("Ground truth success")

    ax = axes[0, 1]
    barplot(ax, df_plans, x="Method", y="Predicted success", hue="Value / Dynamics")
    ax.set_title("Predicted success")

    ax = axes[0, 2]
    barplot(
        ax, df_plans, x="Method", y="Predicted success error", hue="Value / Dynamics"
    )
    ax.set_title("Predicted success error")

    ax = axes[1, 0]
    barplot(ax, df_plans, x="Method", y="Time", hue="Value / Dynamics")
    ax.set_title("Planning time")
    ax.set_ylabel("Time [s]")

    ax = axes[1, 1]
    barplot(ax, df_plans, x="Method", y="Num sampled", hue="Value / Dynamics")
    ax.set_title("Sample quantity")
    ax.set_ylabel("# samples")

    ax = axes[1, 2]
    barplot(
        ax, df_samples, x="Method", y="Predicted success > 0.5", hue="Value / Dynamics"
    )
    ax.set_title("Sample quality")
    ax.set_ylabel("Predicted success > 0.5")

    patches = [
        matplotlib.patches.Patch(color=np.full(3, 0.4 + i * 0.25), label=label)
        for i, label in enumerate(df_plans["Value / Dynamics"].unique()[:-1])
    ]
    axes[0, 2].legend(title="Value / Dynamics", handles=patches, loc="upper right")

    fig.suptitle("Planning results")

    fig.tight_layout()
    fig.savefig(
        pathlib.Path(path) / "planning_results.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=True,
    )
    plt.close(fig)


def plot_action_statistics(
    df_plans: pd.DataFrame, df_samples: pd.DataFrame, path: Union[str, pathlib.Path]
) -> None:
    df_plans = df_plans[
        (df_plans["Value"] != "Oracle")
        & (df_plans["Dynamics"] != "Oracle")
        & (df_plans["Method"] != "Random")
    ]
    df_samples = df_samples[
        (df_samples["Value"] != "Oracle")
        & (df_samples["Dynamics"] != "Oracle")
        & (df_samples["Method"] != "Random")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    sns.kdeplot(
        ax=ax, x="Position", hue="Method", data=df_samples, bw_adjust=0.75, cut=0
    )
    ax.set_title("Sampled Positions")
    ax.set_xlabel("Position [m]")

    ax = axes[0, 1]
    sns.kdeplot(ax=ax, x="Angle", hue="Method", data=df_samples, bw_adjust=0.75, cut=0)
    ax.set_title("Sampled Angles")
    ax.set_xlabel("Angle [rad]")

    ax = axes[1, 0]
    sns.kdeplot(ax=ax, x="Position", hue="Method", data=df_plans, bw_adjust=0.75, cut=0)
    ax.set_title("Planned Positions")
    ax.set_xlabel("Position [m]")

    ax = axes[1, 1]
    sns.kdeplot(ax=ax, x="Angle", hue="Method", data=df_plans, bw_adjust=0.75, cut=0)
    ax.set_title("Planned Angles")
    ax.set_xlabel("Angle [rad]")

    fig.suptitle("Action statistics")

    fig.tight_layout()
    fig.savefig(
        pathlib.Path(path) / "action_statistics.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=True,
    )
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    results = load_results(args.path, args.methods)

    df_plans, df_samples = create_dataframes(results)
    print(df_plans, "\n")
    print(df_samples, "\n")

    plot_planning_results(df_plans, df_samples, args.path)
    plot_action_statistics(df_plans, df_samples, args.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path for output plots")
    parser.add_argument("--methods", nargs="+", help="Method subdirectories")
    args = parser.parse_args()

    main(args)
