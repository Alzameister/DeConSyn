import math
import os

import pandas as pd

from DeConSyn.evaluation.plot import get_runs_path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from DeConSyn.io.io import get_repo_root

def bar_plot(data, baseline_data, out_path, metrics):
    metric_names = []
    means = []
    stds = []
    for metric in metrics:
        metric_data = data[data['Metric'] == metric]
        mean = metric_data['Mean_of_Group_Medians']
        std = metric_data['Std_of_Group_Medians']
        baseline_value = baseline_data[metric].values[0]
        if not np.isnan(baseline_value):
            metric_names.append(metric)
            means.append(mean)
            stds.append(std)

    # Add baseline

    x = np.arange(len(metric_names))
    means = np.array(means).flatten()
    stds = np.array(stds).flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, metric_names, rotation=45, ha='right')
    plt.ylabel("Mean of Group Medians")
    plt.title("Bar Plot of Mean of Group Medians with Std Dev")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

def bar_plot_total(tabddpm_df, baseline_tabddpm_df, ctgan_df, baseline_ctgan_df, out_path, metrics, title):
    plot_data = []
    for metric in metrics:
        tabddpm_metric_data = tabddpm_df[tabddpm_df['Metric'] == metric]
        if tabddpm_metric_data.empty:
            continue
        ctgan_metric_data = ctgan_df[ctgan_df['Metric'] == metric]

        mean_tabddpm = tabddpm_metric_data['Mean_of_Group_Medians'].values[0]
        std_tabddpm = tabddpm_metric_data['Std_of_Group_Medians'].values[0]
        mean_ctgan = ctgan_metric_data['Mean_of_Group_Medians'].values[0]
        std_ctgan = ctgan_metric_data['Std_of_Group_Medians'].values[0]

        plot_data.append({'Metric': metric, 'ModelType': 'DeConSyn-TabDDPM', 'Mean': mean_tabddpm, 'Std': std_tabddpm})
        plot_data.append({'Metric': metric, 'ModelType': 'DeConSyn-CTGAN', 'Mean': mean_ctgan, 'Std': std_ctgan})

        baseline_tabddpm_value = baseline_tabddpm_df[metric].values[0]
        baseline_ctgan_value = baseline_ctgan_df[metric].values[0]
        plot_data.append({'Metric': metric, 'ModelType': 'TabDDPM', 'Mean': baseline_tabddpm_value, 'Std': 0.0})
        plot_data.append({'Metric': metric, 'ModelType': 'CTGAN', 'Mean': baseline_ctgan_value, 'Std': 0.0})

    plot_df = pd.DataFrame(plot_data)

    # 2) Consistent ordering & palette across all small multiples
    order = ['CTGAN', 'DeConSyn-CTGAN', 'TabDDPM', 'DeConSyn-TabDDPM']
    palette = sns.color_palette("pastel")
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(order)}

    # 3) Figure layout
    n_metrics = len(metrics)
    cols = min(3, n_metrics)
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.2 * rows), sharey=False)
    if n_metrics == 1:
        axes = [axes]  # make iterable
    else:
        axes = axes.flatten()

    # 4) Draw each metric’s small plot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data_m = plot_df[plot_df['Metric'] == metric].copy()
        # Ensure consistent order & colors
        data_m['ModelType'] = pd.Categorical(data_m['ModelType'], categories=order, ordered=True)

        sns.barplot(
            data=data_m,
            x='ModelType', y='Mean',
            order=order,
            palette=[color_map[o] for o in order],
            errorbar=None,
            ax=ax
        )

        y_values = data_m['Mean'].values
        y_min = y_values.min() if len(y_values) > 0 else 0
        y_max = y_values.max() if len(y_values) > 0 else 1
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range > 0 else 0.05  # 10% margin or small default
        ax.set_ylim(y_min - margin, y_max + margin)

        # Baseline transparency + value labels
        for p, (_, row) in zip(ax.patches, data_m.sort_values('ModelType').iterrows()):
            # Dim centralized baselines a bit
            if row['ModelType'] in {'CTGAN', 'TabDDPM'}:
                p.set_alpha(0.6)
            h = p.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(p.get_x() + p.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=20)

    # Hide any unused axes
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # 5) Shared legend (top-center)
    # Build from the order & color map so it’s stable even if some metrics miss entries
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color_map[m], label=m, alpha=0.6 if m in {'CTGAN', 'TabDDPM'} else 1.0) for m in
                      order]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(order), frameon=False, title='Model/Type',
               bbox_to_anchor=(0.5, 1.02))

    # 6) Super-title + save
    fig.suptitle(title, y=1.03, fontsize=14, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)

def bar_plot_fedtabdiff(tabddpm_group_stats, fedtabdiff_dfs, out_path, metrics, title):
    # 1) Build long-form dataframe (per metric, per agent_count, per model type)
    agent_counts = sorted(fedtabdiff_dfs.keys(), key=lambda x: int(x))
    plot_data = []
    for metric in metrics:
        for agent_count in agent_counts:
            # DeConSyn aggregation across groups starting with "{n}A"
            mask = (
                tabddpm_group_stats['Group'].str.startswith(f"{agent_count}A") &
                (tabddpm_group_stats['Metric'] == metric)
            )
            group_rows = tabddpm_group_stats[mask]
            if not group_rows.empty:
                medians = group_rows['Median'].astype(float).values
                mean = np.mean(medians)
                std  = np.std(medians, ddof=1) if len(medians) > 1 else 0.0
                plot_data.append({
                    'Metric': metric, 'AgentCount': str(agent_count),
                    'ModelType': 'DeConSyn', 'Mean': mean, 'Std': std
                })

            # FedTabDiff single value per agent count
            fed_val = float(fedtabdiff_dfs[agent_count][metric].values[0])
            plot_data.append({
                'Metric': metric, 'AgentCount': str(agent_count),
                'ModelType': 'FedTabDiff', 'Mean': fed_val, 'Std': 0.0
            })

    plot_df = pd.DataFrame(plot_data)

    # 2) Consistent ordering & palette across all small multiples (match bar_plot_total style)
    order = ['FedTabDiff', 'DeConSyn']  # baseline first, then DeConSyn
    base_palette = sns.color_palette("pastel", n_colors=len(order))
    color_map = {m: base_palette[i] for i, m in enumerate(order)}

    # 3) Figure layout (identical style to bar_plot_total)
    n_metrics = len(metrics)
    cols = min(3, n_metrics)
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.2 * rows), sharey=False)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    # 4) Draw each metric’s small plot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data_m = plot_df[plot_df['Metric'] == metric].copy()
        data_m['ModelType'] = pd.Categorical(data_m['ModelType'], categories=order, ordered=True)

        # We keep AgentCount on x (your FedTabDiff comparison needs it), with hue for ModelType
        sns.barplot(
            data=data_m,
            x='AgentCount', y='Mean',
            hue='ModelType',
            hue_order=order,
            palette=color_map,  # <-- use dict, not list
            errorbar=None,
            dodge=True,
            ax=ax
        )

        y_values = data_m['Mean'].values
        y_min = y_values.min()
        y_max = y_values.max()
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range > 0 else 0.05
        ax.set_ylim(y_min - margin, y_max + margin)

        # Baseline transparency + value labels (same annotation style)
        # Bars are grouped by AgentCount, then by hue order
        # Iterate over bars alongside a sorted dataframe to keep mapping stable
        sorted_rows = data_m.sort_values(['AgentCount', 'ModelType'])
        for p, (_, row) in zip(ax.patches, sorted_rows.iterrows()):
            #if row['ModelType'] == 'FedTabDiff':  # baseline
            #    p.set_alpha(0.6)
            h = p.get_height()
            if not (np.isnan(h) or abs(h) < 1e-9):
                ax.annotate(f"{h:.3f}",
                            xy=(p.get_x() + p.get_width() / 2, h),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(metric)
        ax.set_xlabel('')            # same as total
        ax.set_ylabel('Value')       # same label text as total
        ax.tick_params(axis='x', rotation=20)

        # remove per-axes legends; we’ll add a shared one like in total
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide unused axes
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # 5) Shared legend (top-center), same style as bar_plot_total
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map[m], label=m, alpha=0.6 if m == 'FedTabDiff' else 1.0)
        for m in order
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(order),
               frameon=False, title='Model/Type', bbox_to_anchor=(0.5, 1.02))

    # 6) Super-title + save (identical)
    fig.suptitle(title, y=1.03, fontsize=14, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)

def bar_plot(tabddpm_group_stats, ctgan_group_stats, baseline_tabddpm, baseline_ctgan, fedtabdiff_df, out_path, metrics, title):
    plot_data = []
    # Merge all dfs into one df data

    for metric in metrics:
        metric_mean = tabddpm_group_stats[metric].values[0]
        plot_data.append({'Metric': metric, 'ModelType': 'DeConSyn-TabDDPM', 'Mean': metric_mean})
        metric_mean_ctgan = ctgan_group_stats[metric].values[0]
        plot_data.append({'Metric': metric, 'ModelType': 'DeConSyn-CTGAN', 'Mean': metric_mean_ctgan})
        baseline_value = baseline_tabddpm[metric].values[0]
        plot_data.append({'Metric': metric, 'ModelType': 'TabDDPM', 'Mean': baseline_value})
        baseline_value_ctgan = baseline_ctgan[metric].values[0]
        plot_data.append({'Metric': metric, 'ModelType': 'CTGAN', 'Mean': baseline_value_ctgan})
        fedtabdiff_value = fedtabdiff_df[metric].values[0]
        plot_data.append({'Metric': metric, 'ModelType': 'FedTabDiff', 'Mean': fedtabdiff_value})

    plot_df = pd.DataFrame(plot_data)
    order = ['CTGAN', 'DeConSyn-CTGAN', 'TabDDPM', 'DeConSyn-TabDDPM', 'FedTabDiff']
    palette = sns.color_palette("pastel")
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(order)}

    n_metrics = len(metrics)
    cols = min(3, n_metrics)
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.2 * rows), sharey=False)
    if n_metrics == 1:
        axes = [axes]  # make iterable
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data_m = plot_df[plot_df['Metric'] == metric].copy()
        data_m['ModelType'] = pd.Categorical(data_m['ModelType'], categories=order, ordered=True)

        sns.barplot(
            data=data_m,
            x='ModelType', y='Mean',
            order=order,
            palette=[color_map[o] for o in order],
            errorbar=None,
            ax=ax
        )

        y_values = data_m['Mean'].values
        y_min = y_values.min() if len(y_values) > 0 else 0
        y_max = y_values.max() if len(y_values) > 0 else 1
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range > 0 else 0.05  # 10% margin or small default
        ax.set_ylim(y_min - margin, y_max + margin)

        for p, (_, row) in zip(ax.patches, data_m.sort_values('ModelType').iterrows()):
            if row['ModelType'] in {'CTGAN', 'TabDDPM', 'FedTabDiff'}:
                p.set_alpha(0.6)
            h = p.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(p.get_x() + p.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=20)
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color_map[m], label=m, alpha=0.6 if m in {'CTGAN', 'TabDDPM', 'FedTabDiff'} else 1.0) for m in
                      order]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(order), frameon=False, title='Model/Type',
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.03, fontsize=14, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)


model_type = "tabddpm"
it = 1000
dataset_name = "cardio"
tabddpm_dir = get_runs_path(model_type, dataset_name)
ctgan_dir = get_runs_path("ctgan", dataset_name)
tabddpm_results_dir = os.path.join(tabddpm_dir, "results")
ctgan_results_dir = os.path.join(ctgan_dir, "results")
os.makedirs(tabddpm_results_dir, exist_ok=True)
os.makedirs(ctgan_results_dir, exist_ok=True)

tabddpm_group_stats = pd.read_csv(os.path.join(tabddpm_results_dir, f"{model_type}-{it}-group-stats.csv"))
ctgan_group_stats = pd.read_csv(os.path.join(ctgan_results_dir, f"ctgan-300-group-stats.csv"))
baseline_tabddpm_df = pd.read_csv(os.path.join(tabddpm_dir, f"{model_type}_baseline", "results", "results.csv"))
baseline_ctgan_df = pd.read_csv(os.path.join(ctgan_dir, f"ctgan_baseline", "results", "results.csv"))

repo_root = get_repo_root()
fedtabdiff_dir = repo_root / 'exp' / dataset_name / 'runs' / 'FedTabDiff'
results_4 = pd.read_csv(fedtabdiff_dir / '4A' / 'results.csv')
results_7 = pd.read_csv(fedtabdiff_dir / '7A' / 'results.csv')
results_10 = pd.read_csv(fedtabdiff_dir / '10A' / 'results.csv')
# Merge into one DF
fedtabdiff_dict = {
    "4": results_4,
    "7": results_7,
    "10": results_10
}
metrics = ['LogReg_Accuracy', 'LogReg_F1', 'CatBoost_Accuracy', 'CatBoost_F1', 'DCR', 'NNDR', 'AdversarialAccuracy', 'Disclosure', 'RepU', 'Mean', 'Median', 'Var', 'JS', 'KS']
out_path_total = os.path.join(tabddpm_dir, "plots", f"bar_plot_{dataset_name}_privacy.png")
os.makedirs(os.path.dirname(out_path_total), exist_ok=True)

# Find best fedtabdiff group based on CatBoost_Accuracy
best_fedtabdiff_group = None
best_catboost_acc = -1.0
for agent_count, df in fedtabdiff_dict.items():
    catboost_acc = df['CatBoost_Accuracy'].values[0]
    if catboost_acc > best_catboost_acc:
        best_catboost_acc = catboost_acc
        best_fedtabdiff_group = df

# Find best DeConSyn-TabDDPM group based on CatBoost_Accuracy
best_deconsyn_tabddpm_group = None
best_catboost_acc_deconsyn = -1.0
# Extract rows for CatBoost_Accuracy
mask = (
    (tabddpm_group_stats['Metric'] == 'CatBoost_Accuracy')
)
group_rows = tabddpm_group_stats[mask]
if not group_rows.empty:
    # Sort by Mean column, select best
    group_rows_sorted = group_rows.sort_values(by='Mean', ascending=False)
    top_row = group_rows_sorted.iloc[0]
    catboost_acc = top_row['Mean']
    best_deconsyn_tabddpm_group = top_row['Group']
    best_tabddpm_df = tabddpm_group_stats[tabddpm_group_stats['Group'] == best_deconsyn_tabddpm_group]

best_tabddpm_df = (
    best_tabddpm_df
    .pivot(index="Group", columns="Metric", values="Mean")
    .reset_index()
)

# Find best DeConSyn-CTGAN group based on CatBoost_Accuracy
best_deconsyn_ctgan_group = None
best_catboost_acc_deconsyn_ctgan = -1.0
# Extract rows for CatBoost_Accuracy
mask_ctgan = (
    (ctgan_group_stats['Metric'] == 'CatBoost_Accuracy')
)
group_rows_ctgan = ctgan_group_stats[mask_ctgan]
if not group_rows_ctgan.empty:
    # Sort by Mean column, select best
    group_rows_sorted_ctgan = group_rows_ctgan.sort_values(by='Mean', ascending=False)
    top_row_ctgan = group_rows_sorted_ctgan.iloc[0]
    catboost_acc_ctgan = top_row_ctgan['Mean']
    best_deconsyn_ctgan_group = top_row_ctgan['Group']
    best_ctgan_df = ctgan_group_stats[ctgan_group_stats['Group'] == best_deconsyn_ctgan_group]
best_ctgan_df = (
    best_ctgan_df
    .pivot(index="Group", columns="Metric", values="Mean")
    .reset_index()
)


metrics = ['DCR', 'NNDR', 'AdversarialAccuracy', 'Disclosure', 'RepU']
bar_plot(
    best_tabddpm_df,
    best_ctgan_df,
    baseline_tabddpm_df,
    baseline_ctgan_df,
    best_fedtabdiff_group,
    out_path_total,
    metrics,
    title=f"Comparison of privacy metrics across models ({dataset_name})"
)

metrics = ['LogReg_Accuracy', 'LogReg_F1', 'CatBoost_Accuracy', 'CatBoost_F1']
out_path_total_performance = os.path.join(tabddpm_dir, "plots", f"bar_plot_{dataset_name}_utility.png")
bar_plot(
    best_tabddpm_df,
    best_ctgan_df,
    baseline_tabddpm_df,
    baseline_ctgan_df,
    best_fedtabdiff_group,
    out_path_total_performance,
    metrics,
    title=f"Comparison of utility metrics across models ({dataset_name})"
)

metrics = ['Mean', 'Median', 'Var', 'JS', 'KS', 'CorrelationPearson', 'CorrelationSpearman']
out_path_total_statistical = os.path.join(tabddpm_dir, "plots", f"bar_plot_{dataset_name}_similarity.png")
bar_plot(
    best_tabddpm_df,
    best_ctgan_df,
    baseline_tabddpm_df,
    baseline_ctgan_df,
    best_fedtabdiff_group,
    out_path_total_statistical,
    metrics,
    title=f"Comparison of statistical similarity metrics across models ({dataset_name})"
)



