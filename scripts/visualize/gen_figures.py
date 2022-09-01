import os
from os import path
import json
import argparse
import numpy as np
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt

from temporal_policies.envs.pybox2d.visualization import Box2DVisualizer


def load_planner_results(parent_dir, planner_name):
    fnames = []
    for fn in os.listdir(parent_dir):
        pname = path.splitext(path.basename(fn))[0].split("_")[:-1]
        if "_".join([str(s) for s in pname]) == planner_name:
            fnames.append(fn)

    fpaths = [path.join(parent_dir, fn) for fn in fnames]
    results = []
    for fp in fpaths:
        with open(fp, "r") as fs:
            results.append(json.load(fs))
    return results


def standard_plot(x, y, title, xlabel, ylabel, labels, output_path, std=None):

    for i in range(len(x)):
        plt.plot(
            x[i],
            y[i],
            label=labels[i],
            color=Box2DVisualizer._get_color(i),
            linestyle="-",
            marker="o",
        )
        if std is None:
            continue
        plt.fill_between(
            x[i],
            y[i] - std[i],
            y[i] + std[i],
            color=Box2DVisualizer._get_color(i),
            alpha=0.2,
            linewidth=0,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.close()


def bar_plot(vals, stds, title, xlabel, ylabel, labels, output_path):
    fig, ax = plt.subplots()
    y = np.arange(len(vals))
    ax.barh(
        y,
        vals,
        xerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
        color=[Box2DVisualizer._get_color(i) for i in range(len(vals))],
    )
    ax.set_ylabel(ylabel)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def get_param(result, param_name):
    try:
        param_value = result[param_name]
    except:
        param_value = result["settings"][param_name]
    return param_value


def extract_axes(results, x_name, y_name, y_opt_name=None):
    x = np.zeros(len(results))
    y = np.zeros(len(results))
    y_opt = np.zeros(len(results))
    for i, res in enumerate(results):
        x[i] = get_param(res, x_name)
        y[i] = get_param(res, y_name)
        if y_opt_name is not None:
            y_opt[i] = get_param(res, y_opt_name)
    y_opt = y_opt if y_opt_name is not None else None
    return x, y, y_opt


def extract_sorted_axes(results, x_name, y_name, y_opt_name=None, return_sorted=False):
    x = np.zeros(len(results))
    y = np.zeros(len(results))
    y_opt = np.zeros(len(results))
    for i, res in enumerate(results):
        x[i] = get_param(res, x_name)
        y[i] = get_param(res, y_name)
        if y_opt_name is not None:
            y_opt[i] = get_param(res, y_opt_name)
    sort_idx = x.argsort()
    x = np.sort(x)
    y = y[sort_idx]
    y_opt = y_opt[sort_idx] if y_opt_name is not None else None
    if return_sorted:
        return x, y, y_opt, sort_idx
    return x, y, y_opt


def group_by_param(results, param_names):
    groups = defaultdict(list)
    for res in results:
        if isinstance(param_names, list):
            param_value = tuple(
                [get_param(res, param_name) for param_name in param_names]
            )
        else:
            param_value = get_param(res, param_names)
        groups[param_value].append(deepcopy(res))
    groups_items = list(groups.items())
    return sorted(groups_items, key=lambda x: x[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Path to results directory"
    )
    parser.add_argument(
        "--output-path", "-o", type=str, required=True, help="Path to output figures"
    )
    args = parser.parse_args()

    # Results directories
    easy_path = path.join(args.path, "easy")
    hard_path = path.join(args.path, "hard")

    ## Plot 1: Easy vs Hard - Uniform Sampling
    easy_us_results = load_planner_results(easy_path, "uniform_sampling")
    hard_us_results = load_planner_results(hard_path, "uniform_sampling")
    easy_us_x, easy_us_y, easy_us_std = extract_sorted_axes(
        easy_us_results, "samples", "return_mean", y_opt_name="return_std"
    )
    hard_us_x, hard_us_y, hard_us_std = extract_sorted_axes(
        hard_us_results, "samples", "return_mean", y_opt_name="return_std"
    )

    standard_plot(
        x=[easy_us_x, hard_us_x],
        y=[easy_us_y, hard_us_y],
        std=[easy_us_std**2, hard_us_std**2],
        title="Complexity Analysis: Standard vs Randomized Domain",
        xlabel="Number of Discrete Action Space Partitions",
        ylabel=r"Mean Returns (error $\pm \sigma^2$)",
        labels=["BFS Standard", "BFS Randomized"],
        output_path=path.join(args.output_path, "analysis_env_complexity.png"),
    )

    ## Plot 2-3: Hard - Random Shooting vs Policy Shooting (multiple std)
    for postfix, data_path in zip(["standard", "randomized"], [easy_path, hard_path]):
        rs_results = load_planner_results(data_path, "random_shooting")
        ps_results = load_planner_results(data_path, "policy_shooting")
        rs_x, rs_y, rs_std = extract_sorted_axes(
            rs_results, "samples", "return_mean", y_opt_name="return_std"
        )
        x, y, std, labels = [rs_x], [rs_y], [rs_std], []

        # Store for plot 4
        if postfix == "randomized":
            rs_means = {}
            rs_stds = {}
            for s in [50, 100]:
                rs_means[s] = np.array([rs_y[rs_x == s].item()])
                rs_stds[s] = np.array([rs_std[rs_x == s].item()])

        for standard_deviation, hard_ps_group_results in group_by_param(
            ps_results, "standard_deviation"
        ):
            ps_x, ps_y, ps_std = extract_sorted_axes(
                hard_ps_group_results, "samples", "return_mean", y_opt_name="return_std"
            )
            x.append(ps_x)
            y.append(ps_y)
            std.append(ps_std)
            labels.append(standard_deviation)

            # Store for plot 4
            if standard_deviation == 0.5 and postfix == "randomized":
                ps_means = {}
                ps_stds = {}
                for s in [50, 100]:
                    ps_means[s] = np.array([ps_y[ps_x == s].item()])
                    ps_stds[s] = np.array([ps_std[ps_x == s].item()])

        standard_plot(
            x=x,
            y=y,
            std=None,
            title=f"Primitive init. in Shooting Planners: {postfix.capitalize()} Domain",
            xlabel="Number of Sampled Trajectories",
            ylabel=r"Mean Returns (error $\pm \sigma^2$)",
            labels=["Random Shooting"]
            + [r"Policy Shooting ($\sigma=${:.1f})".format(l) for l in labels],
            output_path=path.join(
                args.output_path, f"analysis_shooting_planners_{postfix}.png"
            ),
        )

    ## Plot 4-5: Hard - Random Shooting vs Policy Shooting (best std) vs CEMs (best std)
    rcem_results = load_planner_results(hard_path, "random_cem")
    pcem_results = load_planner_results(hard_path, "policy_cem")
    rcem_groups = group_by_param(rcem_results, ["standard_deviation", "elites"])
    pcem_groups = group_by_param(pcem_results, ["standard_deviation", "elites"])

    rcem_params, rcem_groups = list(zip(*rcem_groups))
    pcem_params, pcem_groups = list(zip(*pcem_groups))
    assert rcem_params == pcem_params

    for num_trajectories in [50, 100]:
        for (standard_deviation, elites), rcem_group, pcem_group in zip(
            rcem_params, rcem_groups, pcem_groups
        ):
            if elites != 5 or standard_deviation != 0.5:
                continue
            # Extract filter by number of trajectories sampled
            rcem_samples, rcem_iterations, _ = extract_axes(
                rcem_group, "samples", "iterations"
            )
            pcem_samples, pcem_iterations, _ = extract_axes(
                pcem_group, "samples", "iterations"
            )
            rcem_group = [
                r
                for i, r in enumerate(rcem_group)
                if int(rcem_samples[i] * rcem_iterations[i]) == num_trajectories
            ]
            pcem_group = [
                r
                for i, r in enumerate(pcem_group)
                if int(pcem_samples[i] * pcem_iterations[i]) == num_trajectories
            ]
            # Sort returns and return std by increasing samples
            rcem_means, rcem_stds, _ = extract_axes(
                rcem_group, "return_mean", "return_std"
            )
            pcem_means, pcem_stds, _ = extract_axes(
                pcem_group, "return_mean", "return_std"
            )
            rcem_samples, rcem_iterations, _, rcem_sort = extract_sorted_axes(
                rcem_group, "samples", "iterations", return_sorted=True
            )
            pcem_samples, pcem_iterations, _, pcem_sort = extract_sorted_axes(
                pcem_group, "samples", "iterations", return_sorted=True
            )
            rcem_means, rcem_stds = rcem_means[rcem_sort], rcem_stds[rcem_sort]
            pcem_means, pcem_stds = pcem_means[pcem_sort], pcem_stds[pcem_sort]
            rcem_labels = [
                rf"Random CEM ($s={s}$, $i={i}$)"
                for s, i in zip(rcem_samples.astype(int), rcem_iterations.astype(int))
            ]
            pcem_labels = [
                rf"Policy CEM ($s={s}$, $i={i}$)"
                for s, i in zip(pcem_samples.astype(int), pcem_iterations.astype(int))
            ]
            bar_plot(
                vals=np.concatenate(
                    (
                        rs_means[num_trajectories],
                        rcem_means,
                        ps_means[num_trajectories],
                        pcem_means,
                    )
                ),
                stds=np.concatenate((np.zeros(1), rcem_stds, np.zeros(1), pcem_stds)),
                labels=[rf"Random Shooting ($s={num_trajectories}$)"]
                + rcem_labels
                + [rf"Policy Shooting ($s={num_trajectories}$)"]
                + pcem_labels,
                title=rf"Effect of Updating the Sampling Dist. ($\tau={num_trajectories}$, $\sigma={standard_deviation}$, $\epsilon={elites}$)",
                xlabel=r"Mean Returns (error $\pm \sigma$)",
                ylabel="Planner Variations",
                output_path=path.join(
                    args.output_path, f"analysis_cem_{num_trajectories}"
                ),
            )
