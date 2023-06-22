#!/usr/bin/env python3

import pathlib
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import argparse
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


METHOD_IDX = {
    "irl_policy_cem": 0,
    "policy_cem": 1,
    "policy_shooting": 2,
    "daf_gen": 3,
    "random_cem": 4,
    "random_shooting": 5,
    "greedy": 6,
    "daf_skills": 7,
}


def result_indexer(result_data: Tuple[str, str, List[Dict[str, Any]]]) -> int:
    task, method, _ = result_data
    if "daf" in method:
        if task[-1] == method.split("/")[0][-1]:
            return METHOD_IDX["daf_skills"]
        return METHOD_IDX["daf_gen"]
    return METHOD_IDX[method]


def load_results(
    path: Union[str, pathlib.Path],
    envs: Optional[Sequence[Optional[str]]],
    methods: Sequence[str],
    plot_action_statistics: int = 0,
) -> List[Tuple[Optional[str], str, List[Dict[str, Any]]]]:
    if envs is None:
        envs = [None]
    path = pathlib.Path(path)

    results = []
    for task in envs:
        task_results = []
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

            task_results.append((task, method_name, method_results))

        results.extend(sorted(task_results, key=result_indexer))

    return results


def create_dataframe(
    results: List[Tuple[Optional[str], str, List[Dict[str, Any]]]]
) -> pd.DataFrame:
    def get_method_label(method: str, task: str) -> str:
        if method == "random":
            return "Rand."
        if method in ("greedy", "greedy_oracle_dynamics"):
            return "Greedy"

        if "daf" in method:
            train_id, method = method.split("/")
            if task[-1] == train_id[-1]:
                return "DAF Skills"
            return "DAF Gen"

        tokens = method.split("_")
        uq = tokens[0]
        if uq in ["ensemble", "scod"]:
            tokens.pop(0)
            if uq == "ensemble":
                uq = "Ens."
            elif uq == "scod":
                uq = "SCOD"
        else:
            uq = None

        policy = tokens[0]
        if policy == "random":
            policy = "Rand."
        elif policy == "policy":
            policy = "Policy"
        elif policy == "irl":
            policy = "IRL Policy"
            tokens = tokens[1:]

        planner = tokens[1]
        if planner == "cem":
            planner = "CEM"
        elif planner == "shooting":
            planner = "Shoot."

        if len(tokens) > 2:
            for t in tokens[2:]:
                p = t.split("-")
                planner = f"{planner} {p[0][0].upper()}={p[-1]}"

        if uq is None:
            return f"{policy} {planner}"

        return f"{uq} {policy} {planner}"

    def get_task_label(task: Optional[str]) -> str:
        if task is None:
            return ""

        tokens = task.split("/")
        task_name = f"Task {tokens[-1][-1]}"
        assert len(tokens) == 2
        domain_name = " ".join([w.capitalize() for w in tokens[0].split("_")])
        return f"{domain_name}: {task_name}"
        
    df_plans: Dict[str, List[Any]] = {
        "Task": [],
        "Method": [],
        "Task Success": [],
        "Success Type": [],
    }

    for task, method, method_results in results:
        task_label = get_task_label(task)
        method_label = get_method_label(method, task)
        for result in method_results:
            df_plans["Task"].append(task_label)
            df_plans["Method"].append(method_label)
            df_plans["Task Success"].append(
                0.0 if method == "greedy" else result["p_success"].item()
            )
            df_plans["Success Type"].append("Predicted success")

            df_plans["Task"].append(task_label)
            df_plans["Method"].append(method_label)
            df_plans["Task Success"].append(result["rewards"].prod())
            df_plans["Success Type"].append("Ground truth success")

            df_plans["Task"].append(task_label)
            df_plans["Method"].append(method_label)
            df_plans["Task Success"].append(
                result["rewards"].sum() / len(result["rewards"])
            )
            df_plans["Success Type"].append("Sub-goal completion")

    return pd.DataFrame(df_plans)


def plot_planning_results(
    df_plans: pd.DataFrame,
    path: Union[str, pathlib.Path],
    name: str,
) -> None:
    palette = (
        sns.color_palette()[:5]
        + sns.color_palette()[7:8]
        + sns.color_palette()[6:7]
        + sns.color_palette()[8:]
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
            # except for the second one, which is NA / NA.
            if idx_success_type == 0:
                a = 0.8
            elif idx_success_type == 1:
                a = 0.0
                if idx_class == num_classes - 1:
                    a = 0.45
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

            if idx_success_type == 1 and idx_class == num_classes - 1:
                bar.set_hatch("//")
                bar.set_edgecolor("w")

            if idx_success_type == 2 and idx_class != num_classes - 2:
                bar.set_fill(False)
                bar.set_linewidth(2.7)
                bar.set_linestyle(":")
                bar.set_y(bar.get_height())
                bar.set_height(
                    ground_truth_bars[idx_class].get_height() - bar.get_height()
                )
            line.set_alpha(0)

    tasks = df_plans["Task"].unique()
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), dpi=300)

    for idx_task, task in enumerate(tasks):
        ax = axes.flatten()[idx_task]
        barplot(
            ax=ax,
            df_plans=df_plans[df_plans["Task"] == task],
            x="Method",
            y="Task Success",
            hue="Success Type",
            ylim=(0, 1),
        )
        ax.set_title(task[:-1] + str(int(task[-1]) + 1))
        ax.set_ylabel("")

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
    axes[0, -1].legend(handles=patches, loc="upper right")

    path = pathlib.Path(path)

    fig.tight_layout()
    fig.savefig(
        f"plots/{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=True,
    )


def aggregate_metric(
    df_plans: pd.DataFrame, method_name: str, success_type: str
) -> float:
    method_df = df_plans[df_plans["Method"] == method_name]
    method_res = method_df[method_df["Success Type"] == success_type]
    return method_res["Task Success"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path for output plots.")
    parser.add_argument("--envs", nargs="+", help="Planning domain subdirectories.")
    parser.add_argument("--methods", nargs="+", help="Method subdirectories.")
    parser.add_argument("--name", default="test", help="Figure filename.")
    args = parser.parse_args()

    results = load_results(args.path, args.envs, args.methods)
    df_plans = create_dataframe(results)
    plot_planning_results(df_plans, args.path, args.name)
    for method_name in df_plans["Method"].unique():
        print(f"--- Method: {method_name} ---")
        print(
            f"Ground truth success: {aggregate_metric(df_plans, method_name, 'Ground truth success')}"
        )
