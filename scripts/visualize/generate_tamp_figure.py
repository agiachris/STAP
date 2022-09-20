#!/usr/bin/env python3

import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_results(
    path: Union[str, pathlib.Path],
    envs: Optional[Sequence[Optional[str]]],
    methods: Sequence[str],
    plot_action_statistics: int = 0,
) -> Generator[Tuple[Optional[str], str, List[Dict[str, Any]]], None, None]:
    if envs is None:
        envs = [None]
    path = pathlib.Path(path)

    for task in envs:
        env_path = path if task is None else path / task
        for method_name in methods:
            method_results = []
            for npz_file in tqdm.tqdm(
                sorted(
                    (env_path / method_name).glob("results_*.npz"),
                    key=lambda x: int(x.stem.split("_")[-1]),
                )
            ):
                with open(npz_file, "rb") as f:
                    npz = np.load(f, allow_pickle=True)
                    d = {
                        "scaled_actions": npz["scaled_actions"],
                        "p_success": npz["p_success"],
                        "values": npz["values"],
                        "rewards": npz["rewards"],
                        "p_visited_success": npz["p_visited_success"],
                        "t_planner": npz["t_planner"],
                    }
                    if plot_action_statistics:
                        d["scaled_visited_actions"] = npz["scaled_visited_actions"]
                    method_results.append(d)

                # print(npz_file, results[method_name][-1]["rewards"])
            yield task, method_name, method_results


def create_dataframe(
    results: Generator[Tuple[Optional[str], str, List[Dict[str, Any]]], None, None]
) -> pd.DataFrame:
    def get_method_label(method: str) -> str:
        if method == "random":
            return "Rand."
        if method in ("greedy", "greedy_oracle_dynamics"):
            return "Greedy"

        tokens = method.split("_")
        if tokens[0] == "daf":
            return "DAF Skills"
        elif "daf" in tokens[0]:
            return "DAF Gen"

        policy = tokens[0]
        planner = tokens[1]
        if policy == "scod":
            policy = f"SCOD {planner.capitalize()}"
            planner = "CEM"
        elif policy == "random":
            policy = "Rand."
        else:
            policy = policy.capitalize()
        if planner == "cem":
            planner = planner.upper()
        elif planner == "shooting":
            planner = "Shoot."
        if "oracle" in tokens:
            if "value" in tokens:
                planner = f"{planner} oracle val/dyn"
            else:
                planner = f"{planner} oracle dyn"
        print(policy, planner, tokens)
        return f"{policy} {planner}"

    def get_task_label(task: Optional[str]) -> str:
        if task is None:
            return ""

        tokens = task.split("/")
        domain = " ".join([w.capitalize() for w in tokens[0].split("_")])

        return f"{domain}: TAMP Task"

    df_plans: Dict[str, List[Any]] = {
        "Task": [],
        "Method": [],
        "Task Success": [],
        "Success Type": [],
        "Planning Time": [],
    }

    for task, method, method_results in results:
        task_label = get_task_label(task)
        method_label = get_method_label(method)
        for result in method_results:
            df_plans["Task"].append(task_label)
            df_plans["Method"].append(method_label)
            df_plans["Task Success"].append(
                0.0 if method == "greedy" else result["p_success"].item()
            )
            df_plans["Success Type"].append("Predicted success")
            df_plans["Planning Time"].append(result["t_planner"][0])

            df_plans["Task"].append(task_label)
            df_plans["Method"].append(method_label)
            df_plans["Task Success"].append(result["rewards"].prod())
            df_plans["Success Type"].append("Ground truth success")
            df_plans["Planning Time"].append(result["t_planner"][0])

            df_plans["Task"].append(task_label)
            df_plans["Method"].append(method_label)
            df_plans["Task Success"].append(
                result["rewards"].sum() / len(result["rewards"])
            )
            df_plans["Success Type"].append("Sub-goal completion")
            df_plans["Planning Time"].append(result["t_planner"][0])

    return pd.DataFrame(df_plans)


def plot_planning_results(
    df_plans: pd.DataFrame,
    path: Union[str, pathlib.Path],
) -> None:
    palette = (
        [sns.color_palette()[-1]]
        + sns.color_palette()[:2]
        + sns.color_palette()[3:5]
        + sns.color_palette()[6:]
    )

    def barplot(
        ax: plt.Axes,
        df_plans: pd.DataFrame,
        ylim: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> plt.Axes:
        idx_subgoal = df_plans["Success Type"] == "Sub-goal completion"
        idx_ground_truth = df_plans["Success Type"] == "Ground truth success"
        idx_predicted = df_plans["Success Type"] == "Predicted success"
        sns.barplot(ax=ax, data=df_plans[idx_subgoal], **kwargs)
        sns.barplot(ax=ax, data=df_plans[idx_ground_truth], **kwargs)
        sns.barplot(ax=ax, data=df_plans[idx_predicted], **kwargs)
        ax.set_xticklabels(
            [label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()]
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("")

        # Change colors and shift bars.
        num_classes = len(df_plans[kwargs["x"]].unique())
        ground_truth_bars = []
        for idx_bar, (bar, line) in enumerate(zip(ax.patches, ax.lines)):
            idx_class = idx_bar % num_classes
            idx_success_type = idx_bar // num_classes

            # Compute color.
            # Colors should increase in lightness with idx_success_type,
            # except for the last one, which is NA / NA.
            if idx_success_type == 0:
                a = 0.8
            elif idx_success_type == 1:
                a = 0.0
                ground_truth_bars.append(bar)
            else:
                a = 0.5

            color = 0.8 * np.array(palette[idx_class])
            color = (1 - a) * color + a * np.ones(3)

            # Compute position.
            # A column will either contain NA / NA or all the other
            # value/dynamics variants. Since NA / NA is the only bar in its
            # column and is by default placed on the right side of the column,
            # it should be shifted left to center it. All the other variants
            # should be shifted right to make up for the gap left by the missing
            # NA / NA bar.
            if idx_success_type == 2:
                dx = -2
            else:
                dx = 0
            dx *= bar.get_width()

            # Modify plot.
            bar.set_color(color)

            if idx_success_type == 2 and idx_class != num_classes - 1:
                bar.set_fill(False)
                bar.set_linewidth(2.7)
                bar.set_linestyle(":")
                bar.set_y(bar.get_height())
                bar.set_height(
                    ground_truth_bars[idx_class].get_height() - bar.get_height()
                )
            line.set_alpha(0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.75), dpi=300)
    # ax = axes[0]
    # barplot(
    #     ax=ax,
    #     df_plans=df_plans,
    #     x="Method",
    #     y="Task Success",
    #     hue="Success Type",
    #     ylim=[0, 1],
    # )
    # ax.set_title("Average")

    ax = axes[0]
    barplot(
        ax=ax,
        df_plans=df_plans[df_plans["Task"] == "Hook Reach: TAMP Task"],
        x="Method",
        y="Task Success",
        hue="Success Type",
        ylim=(0, 1),
    )
    ax.set_ylabel("")
    ax.set_title("Hook Reach: TAMP Problem")

    ax = axes[1]
    barplot(
        ax=ax,
        df_plans=df_plans[df_plans["Task"] == "Constrained Packing: TAMP Task"],
        x="Method",
        y="Task Success",
        hue="Success Type",
        ylim=(0, 1),
    )
    ax.set_ylabel("")
    ax.set_title("Constrained Packing: TAMP Problem")

    def get_alpha(idx_success_type: int) -> float:
        if idx_success_type == 0:
            a = 0.8
        elif idx_success_type == 1:
            a = 0.0
        else:
            a = 0.5
        return a

    patches = [
        matplotlib.patches.Patch(
            color=np.full(3, 0.0),
            label="Task success",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            color=np.full(3, 0.5),
            linestyle=":",
            linewidth=2.7,
            label="Predicted task success",
        ),
        matplotlib.patches.Patch(
            color=np.full(3, 0.8),
            label="Sub-goal completion",
        ),
    ]
    axes[0].legend(handles=patches, loc="lower left")

    path = pathlib.Path(path)

    fig.tight_layout()
    fig.savefig(
        "6-tamp.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=True,
    )
    # plt.close(fig)


if __name__ == "__main__":
    path = "../../plots/20220914/official/tamp_experiment"
    envs = ["hook_reach/tamp0", "constrained_packing/tamp0"]
    methods = [
        "scod_policy_cem",
        "policy_cem",
        "policy_shooting",
        "train0/daf_random_cem",
        "train1/daf_random_cem",
        "train2/daf_random_cem",
        "random_cem",
        "random_shooting",
        "greedy",
    ]
    results = load_results(path, envs, methods)

    df_plans = create_dataframe(results)
    plot_planning_results(df_plans, path)
