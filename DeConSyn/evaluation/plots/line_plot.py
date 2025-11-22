import numpy as np
import os
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from DeConSyn.evaluation.plot import get_runs_path


def line_plot_vs_topology_by_dataset(group_stats,
                                     metrics_to_plot,
                                     path,
                                     title,
                                     topology_order,
                                     dataset_order=None,
                                     dataset_labels=None,
                                     add_trend: bool = False):
    """
    For each topology, aggregate over ALL Num Agents and plot
    one point per topology, with one line per dataset.

    Aggregation:
      - For each (Dataset, topology, Metric), we take:
          Mean_agg = mean of 'Mean' across all Num Agents
          Std_agg  = std of 'Mean' across all Num Agents

    Additionally, we save this aggregated table as a CSV
    next to the figure (same basename + '-agg.csv').
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        sharex=True,
    )
    axes = axes.flatten() if n_metrics > 1 else [axes]

    if dataset_order is None:
        dataset_order = sorted(group_stats["Dataset"].unique())
    if dataset_labels is None:
        dataset_labels = {d: d.capitalize() for d in dataset_order}

    topo_to_x = {t: i for i, t in enumerate(topology_order)}
    x_ticks = list(range(len(topology_order)))

    # ---- aggregate ONCE for this plot ----
    agg_all = (
        group_stats
        .groupby(["Dataset", "topology", "Metric"], as_index=False)
        .agg(
            Mean_agg=("Mean", "mean"),
            Std_agg=("Mean", "std"),
        )
    )

    # save aggregated table next to the figure
    agg_out_path = os.path.splitext(path)[0] + "-agg.csv"
    agg_all.to_csv(agg_out_path, index=False)

    # --- main plotting loop ---
    for ax, (metric, metric_title) in zip(axes, metrics_to_plot):
        # filter aggregated table for this metric
        agg = agg_all[agg_all["Metric"] == metric].copy()

        for ds in dataset_order:
            g = agg[agg["Dataset"] == ds].copy()
            if g.empty:
                continue

            g["x"] = g["topology"].map(topo_to_x)
            g = g.dropna(subset=["x"]).sort_values("x")

            # main errorbar line
            ax.errorbar(
                g["x"],
                g["Mean_agg"],
                yerr=g["Std_agg"],
                marker="o",
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2,
                label=dataset_labels.get(ds, ds),
            )

            # optional trend
            if add_trend and len(g) >= 2:
                x = g["x"].to_numpy(dtype=float)
                y = g["Mean_agg"].to_numpy(dtype=float)

                a, b = np.polyfit(x, y, deg=1)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = a * x_fit + b

                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )

        ax.set_xlabel("Topology", fontsize=10)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(topology_order, rotation=0)

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    # y-labels for left column
    for r in range(n_rows):
        idx = r * n_cols
        if idx < n_metrics:
            axes[idx].set_ylabel("Value", fontsize=10)
    for ax in axes[:n_metrics]:
        ax.tick_params(axis="x", which="both", labelbottom=True)

    # legend
    handles, labels = [], []
    for ax in axes[:n_metrics]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    if handles:
        fig.legend(
            handles,
            labels,
            title="Dataset",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=len(handles),
            frameon=True,
            fontsize=10,
            title_fontsize=11,
        )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    sns.reset_defaults()


def line_plot_vs_agents_by_dataset_all_topologies(group_stats,
                                                  metrics_to_plot,
                                                  path,
                                                  title,
                                                  dataset_order=None,
                                                  dataset_labels=None,
                                                  add_trend: bool = False):
    """
    For each number of agents, aggregate over ALL topologies and plot
    one point per agent count, with one line per dataset.

    Aggregation:
      - For each (Dataset, Num Agents, Metric), we take:
          Mean_agg = mean of 'Mean' across all topologies
          Std_agg  = std of 'Mean' across all topologies

    Additionally, we save this aggregated table as a CSV
    next to the figure (same basename + '-agg.csv').
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        sharex=True,
    )
    axes = axes.flatten() if n_metrics > 1 else [axes]

    if dataset_order is None:
        dataset_order = sorted(group_stats["Dataset"].unique())
    if dataset_labels is None:
        dataset_labels = {d: d.capitalize() for d in dataset_order}

    # ---- aggregate ONCE for this plot ----
    agg_all = (
        group_stats
        .groupby(["Dataset", "Num Agents", "Metric"], as_index=False)
        .agg(
            Mean_agg=("Mean", "mean"),
            Std_agg=("Mean", "std"),
        )
    )

    # save aggregated table next to the figure
    agg_out_path = os.path.splitext(path)[0] + "-agg.csv"
    agg_all.to_csv(agg_out_path, index=False)

    for ax, (metric, metric_title) in zip(axes, metrics_to_plot):
        # filter aggregated table for this metric
        agg = agg_all[agg_all["Metric"] == metric].copy()

        for ds in dataset_order:
            g = agg[agg["Dataset"] == ds].copy()
            if g.empty:
                continue

            g = g.sort_values("Num Agents")

            # main errorbar line
            ax.errorbar(
                g["Num Agents"],
                g["Mean_agg"],
                yerr=g["Std_agg"],
                marker="o",
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2,
                label=dataset_labels.get(ds, ds),
            )

            # optional trend
            if add_trend and len(g) >= 2:
                x = g["Num Agents"].to_numpy(dtype=float)
                y = g["Mean_agg"].to_numpy(dtype=float)

                a, b = np.polyfit(x, y, deg=1)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = a * x_fit + b

                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )

        ax.set_xlabel("Number of agents $n$", fontsize=10)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        agent_values = sorted(agg_all["Num Agents"].unique())
        ax.set_xticks(agent_values)
        ax.set_xticklabels(agent_values)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    # y-labels for left column
    for r in range(n_rows):
        idx = r * n_cols
        if idx < n_metrics:
            axes[idx].set_ylabel("Value", fontsize=10)

    for ax in axes[:n_metrics]:
        ax.tick_params(axis="x", which="both", labelbottom=True)

    # legend
    handles, labels = [], []
    for ax in axes[:n_metrics]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break


    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    if handles:
        fig.legend(
            handles,
            labels,
            title="Dataset",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=len(handles),
            frameon=True,
            fontsize=10,
            title_fontsize=11,
        )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    sns.reset_defaults()



def load_group_stats_for_model(model_type: str) -> pd.DataFrame:
    """
    Load and combine group-stats for all datasets for a given model type.
    Adds:
      - Num Agents
      - E (epochs)
      - R (iterations)
      - topology
      - Dataset
    """
    frames = []

    for ds in DATASETS:
        model_dir = get_runs_path(model_type, ds)
        results_dir = os.path.join(model_dir, "results")
        csv_name = GROUP_STATS_FILES[model_type]
        csv_path = os.path.join(results_dir, csv_name)

        df = pd.read_csv(csv_path)

        # Parse "Group" like: "A4 1E 300R ring"
        parts = df["Group"].str.split(" ", expand=True)
        df["Num Agents"] = parts[0].str.replace("A", "", regex=False).astype(int)
        df["E"] = parts[1].str.replace("E", "", regex=False).astype(int)
        df["R"] = parts[2].str.replace("R", "", regex=False).astype(int)
        df["topology"] = parts[3]

        # Mark which dataset this row comes from
        df["Dataset"] = ds

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined


#dataset_names = ['adult', 'cardio', 'churn']
DATASETS = ["adult", "cardio", "churn"]
# Filenames for your group-stats per model
GROUP_STATS_FILES = {
    "tabddpm": "tabddpm-1000-group-stats.csv",
    "ctgan": "ctgan-300-group-stats.csv",
}
metrics = [
    ("DCR", "DCR"),
    ("NNDR", "NNDR"),
    ("AdversarialAccuracy", "Adversarial Accuracy"),
    ("RepU", "RepU"),
    ("Disclosure", "Disclosure"),
    ("Mean", "Mean"),
    ("Var", "Var"),
    ("Median", "Median"),
    ("JS", "JS"),
    ("KS", "KS"),
    ("CorrelationSpearman", "Correlation Spearman"),
    ("CorrelationPearson", "Correlation Pearson"),
    ("CatBoost_Accuracy", "CatBoost Accuracy"),
    ("CatBoost_F1", "CatBoost F1 Score"),
]
dataset_labels = {"adult": "Adult", "cardio": "Cardio", "churn": "Churn"}
topology_order = ["ring", "smallworld", "full"]

group_stats_tabddpm_all = load_group_stats_for_model("tabddpm")
group_stats_ctgan_all = load_group_stats_for_model("ctgan")

metrics = [
    ("DCR", "DCR"),
    ("NNDR", "NNDR"),
    ("AdversarialAccuracy", "Adversarial Accuracy"),
    ("RepU", "RepU"),
    ("Disclosure", "Disclosure")
]

line_plot_vs_topology_by_dataset(
    group_stats_tabddpm_all,
    metrics,
    os.path.join(get_runs_path("tabddpm", "adult"), "tabddpm-line-plot-vs-topology-privacy.svg"),
    "DeConSyn-TabDDPM – Effect of topology by dataset on privacy metrics",
    topology_order=topology_order,
    dataset_labels=dataset_labels,
)

line_plot_vs_agents_by_dataset_all_topologies(
    group_stats_tabddpm_all,
    metrics,
    os.path.join(get_runs_path("tabddpm", "adult"), "tabddpm-line-plot-vs-agents-privacy.svg"),
    "DeConSyn-TabDDPM – Effect of number of agents by dataset on privacy metrics",
    dataset_labels=dataset_labels,
)

line_plot_vs_topology_by_dataset(
    group_stats_ctgan_all,
    metrics,
    os.path.join(get_runs_path("ctgan", "adult"), "ctgan-line-plot-vs-topology-privacy.svg"),
    "DeConSyn-CTGAN – Effect of topology by dataset on privacy metrics",
    topology_order=topology_order,
    dataset_labels=dataset_labels,
)

line_plot_vs_agents_by_dataset_all_topologies(
    group_stats_ctgan_all,
    metrics,
    os.path.join(get_runs_path("ctgan", "adult"), "ctgan-line-plot-vs-agents-privacy.svg"),
    "DeConSyn-CTGAN – Effect of number of agents by dataset on privacy metrics",
    dataset_labels=dataset_labels,
)


metrics = [
    ("Mean", "Mean"),
    ("Var", "Var"),
    ("Median", "Median"),
    ("JS", "JS"),
    ("KS", "KS"),
    ("CorrelationSpearman", "Correlation Spearman"),
    ("CorrelationPearson", "Correlation Pearson"),
]

line_plot_vs_topology_by_dataset(
    group_stats_tabddpm_all,
    metrics,
    os.path.join(get_runs_path("tabddpm", "adult"), "tabddpm-line-plot-vs-topology-similarity.svg"),
    "DeConSyn-TabDDPM – Effect of topology by dataset on similarity metrics",
    topology_order=topology_order,
    dataset_labels=dataset_labels,
)

line_plot_vs_agents_by_dataset_all_topologies(
    group_stats_tabddpm_all,
    metrics,
    os.path.join(get_runs_path("tabddpm", "adult"), "tabddpm-line-plot-vs-agents-similarity.svg"),
    "DeConSyn-TabDDPM – Effect of number of agents by dataset on similarity metrics",
    dataset_labels=dataset_labels,
)

line_plot_vs_topology_by_dataset(
    group_stats_ctgan_all,
    metrics,
    os.path.join(get_runs_path("ctgan", "adult"), "ctgan-line-plot-vs-topology-similarity.svg"),
    "DeConSyn-CTGAN – Effect of topology by dataset on similarity metrics",
    topology_order=topology_order,
    dataset_labels=dataset_labels,
)

line_plot_vs_agents_by_dataset_all_topologies(
    group_stats_ctgan_all,
    metrics,
    os.path.join(get_runs_path("ctgan", "adult"), "ctgan-line-plot-vs-agents-similarity.svg"),
    "DeConSyn-CTGAN – Effect of number of agents by dataset on similarity metrics",
    dataset_labels=dataset_labels,
)

metrics = [
    ("LogReg_Accuracy", "Logistic Regression Accuracy"),
    ("LogReg_F1", "Logistic Regression F1 Score"),
    ("CatBoost_Accuracy", "CatBoost Accuracy"),
    ("CatBoost_F1", "CatBoost F1 Score")
]

line_plot_vs_topology_by_dataset(
    group_stats_tabddpm_all,
    metrics,
    os.path.join(get_runs_path("tabddpm", "adult"), "tabddpm-line-plot-vs-topology-utility.svg"),
    "DeConSyn-TabDDPM – Effect of topology by dataset on utility metrics",
    topology_order=topology_order,
    dataset_labels=dataset_labels,
)

line_plot_vs_agents_by_dataset_all_topologies(
    group_stats_tabddpm_all,
    metrics,
    os.path.join(get_runs_path("tabddpm", "adult"), "tabddpm-line-plot-vs-agents-utility.svg"),
    "DeConSyn-TabDDPM – Effect of number of agents by dataset on utility metrics",
    dataset_labels=dataset_labels,
)

line_plot_vs_topology_by_dataset(
    group_stats_ctgan_all,
    metrics,
    os.path.join(get_runs_path("ctgan", "adult"), "ctgan-line-plot-vs-topology-utility.svg"),
    "DeConSyn-CTGAN – Effect of topology by dataset on utility metrics",
    topology_order=topology_order,
    dataset_labels=dataset_labels,
)

line_plot_vs_agents_by_dataset_all_topologies(
    group_stats_ctgan_all,
    metrics,
    os.path.join(get_runs_path("ctgan", "adult"), "ctgan-line-plot-vs-agents-utility.svg"),
    "DeConSyn-CTGAN – Effect of number of agents by dataset on utility metrics",
    dataset_labels=dataset_labels,
)