import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import permutation_test

from DeConSyn.evaluation.plot import get_avg_agents_df, get_baseline_df, get_runs_path, get_fedtabdiff_df

METRICS = ['DCR', 'NNDR', 'AdversarialAccuracy', 'JS', 'KS', 'RepU', 'DiSCO', 'Mean', 'Median', 'Var', 'LogReg_Accuracy', 'LogReg_F1']
METRICS_BETTER = {
    'DCR': 'Higher',
    'NNDR': 'Higher',
    'AdversarialAccuracy': 'Higher',
    'RepU': 'Lower',
    'DiSCO': 'Lower',
    'Var': "Lower",
    'Mean': "Lower",
    'Median': "Lower",
    'JS': 'Higher',
    'KS': 'Higher',
    'CorrelationPearson': 'Higher',
    'CorrelationSpearman': 'Higher',
    'LogReg_Accuracy': 'Higher',
    'LogReg_F1': 'Higher',
    'Disclosure': 'Lower'
}

def extract_group(run_name):
    # Extract everything after the first integer and dash
    match = re.search(r'(\d+)Agents-(\d+)Epochs-(\d+)Iterations-(\w+)', run_name)
    if match:
        agents, epochs, iterations, topology = match.groups()
        return f"{agents}A {epochs}E {iterations}R {topology}"
    match = re.search(r'(\d+)Agents-(\d+)Epochs-(\d+)Rounds-(\w+)', run_name)
    if match:
        agents, epochs, iterations, topology = match.groups()
        return f"{agents}A {epochs}E {iterations}R {topology}"
    return run_name

def paired_t_ci(diff, alpha=0.05):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean = diff.mean()
    sd = diff.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean - tcrit * se
    ci_upper = mean + tcrit * se
    t_stat, p_value = stats.ttest_1samp(diff, popmean=0)
    return mean, ci_lower, ci_upper, p_value

def compare(model_type, it, dataset_name="adult"):
    runs_dir = get_runs_path(model_type, dataset_name)
    runs_results_dir = os.path.join(runs_dir, "results")
    os.makedirs(runs_results_dir, exist_ok=True)
    runs_df: pd.DataFrame = get_avg_agents_df(it, model_type, dataset_name)
    runs_df.to_csv(os.path.join(runs_results_dir, f"{model_type}-{it}.csv"))
    baseline_df = get_baseline_df(model_type, dataset_name)
    numeric_cols = runs_df.select_dtypes(include='number').columns
    if runs_df is None or runs_df.empty:
        print("No data available for analysis.")
        raise SystemExit

    runs_df_with_group = runs_df.copy()
    runs_df_with_group['Group'] = runs_df_with_group['run'].apply(extract_group)

    group_stats = []
    for group, grp_df in runs_df_with_group.groupby('Group'):
        for col in numeric_cols:
            vals = grp_df[col].dropna().values
            if vals.size == 0:
                continue
            mean = float(np.mean(vals))
            std = vals.std()
            median = float(np.median(vals))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            iqr = q3 - q1
            group_stats.append({
                "Group": group,
                "Metric": col,
                "Mean": mean,
                "Std": std,
                "Median": median,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr
            })
    if group_stats:
        group_stats_df = pd.DataFrame(group_stats)
        # Optional stable ordering similar to sample: sort by Group then Metric
        group_stats_df = group_stats_df.sort_values(["Group", "Metric"]).reset_index(drop=True)
        group_stats_df.to_csv(os.path.join(runs_results_dir, f"{model_type}-{it}-group-stats.csv"), index=True)
    # --- end new section ---

    methods = runs_df[runs_df['run'] != 'baseline_ctgan']
    group_diffs = {}
    for method in methods['run'].unique():
        group_id = extract_group(method)
        method_data = methods[methods['run'] == method]
        method_diffs = {}
        for col in numeric_cols:
            if col in baseline_df.columns and col in method_data.columns:
                diff = method_data[col].values - baseline_df[col].values
                method_diffs.setdefault(col, []).append(diff)
        if group_id not in group_diffs:
            group_diffs[group_id] = {}
        for col, diff_list in method_diffs.items():
            group_diffs[group_id].setdefault(col, []).extend(diff_list)

    results = []
    for group_id, metrics in group_diffs.items():
        for col, diffs_list in metrics.items():
            all_diffs = np.concatenate(diffs_list)
            # Paired t-test CI
            mean_t, lower_t, upper_t, p_value_t = paired_t_ci(all_diffs)
            # Permutation CI
            results.append({
                "Group": group_id,
                "Metric": col,
                "Mean_t": mean_t,
                "CI_Lower_t": lower_t,
                "CI_Upper_t": upper_t,
                "p_value_t": p_value_t,
            })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(runs_results_dir, f"{model_type}-{it}-ci.csv"))
    run_dir = get_runs_path(model_type)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for metric in df["Metric"].unique():
        if metric == 'AdversarialAccuracy':
            print(metric)
        subset = df[df["Metric"] == metric].sort_values("Group", ascending=False)
        if subset["Mean_t"].notna().sum() == 0:
            continue
        subset = subset.dropna(subset=["Group", "Mean_t", "CI_Lower_t", "CI_Upper_t"])
        subset["Group"] = subset["Group"].astype(str)
        colors_t = np.where(subset["p_value_t"] < 0.05, "red", "tab:blue")
        plt.figure(figsize=(14, 7))
        # Paired t-test CI (gray)
        plt.errorbar(
            subset["Group"], subset["Mean_t"],
            yerr=[subset["Mean_t"] - subset["CI_Lower_t"], subset["CI_Upper_t"] - subset["Mean_t"]],
            fmt='o', capsize=7, markersize=8, color='tab:gray', ecolor='tab:gray', label="t-test CI"
        )
        plt.scatter(
            subset["Group"], subset["Mean_t"],
            c=colors_t, s=80, zorder=3, label="t-test Significant (p<0.05)"
        )
        for i, group in enumerate(subset["Group"]):
            # Get all individual differences for this group/metric
            diffs = np.concatenate(group_diffs[group][metric])
            plt.scatter([group] * len(diffs), diffs, color='black', alpha=0.3, s=30, zorder=2,
                        label="Individual diffs" if i == 0 else None)
        plt.title(
            f"{metric}: Mean Diff with 95% CI (t-test) ({METRICS_BETTER[metric] if metric in METRICS_BETTER else 'N/A'} = better)",
            fontsize=18)
        plt.ylabel("Mean Difference", fontsize=14)
        plt.xlabel("Group", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{run_dir}/plots/ci_{metric}_t.png")
        plt.close()

    metrics = [m for m in df["Metric"].unique() if df[df["Metric"] == m]["Mean_t"].notna().sum() > 0]
    if metrics:
        import math
        n = len(metrics)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
        for idx, metric in enumerate(metrics):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            subset = df[df["Metric"] == metric].sort_values("Group", ascending=False)
            subset = subset.dropna(subset=["Group", "Mean_t", "CI_Lower_t", "CI_Upper_t"])
            if subset.empty:
                ax.axis('off')
                continue
            colors_t = np.where(subset["p_value_t"] < 0.05, "red", "tab:blue")
            groups = subset["Group"].astype(str).tolist()
            x_pos = np.arange(len(groups))
            # Errorbars (t-test CI)
            y = subset["Mean_t"].values
            yerr = np.vstack([y - subset["CI_Lower_t"].values, subset["CI_Upper_t"].values - y])
            ax.errorbar(x_pos, y, yerr=yerr, fmt='o', capsize=5, markersize=6,
                        color='tab:gray', ecolor='tab:gray', label="t-test CI")
            ax.scatter(x_pos, y, c=colors_t, s=60, zorder=3, label="t-test Significant (p<0.05)")
            # Individual diffs without jitter (aligned to group x position)
            for i, group in enumerate(groups):
                if group in group_diffs and metric in group_diffs[group]:
                    diffs = np.concatenate(group_diffs[group][metric])
                    ax.scatter(np.full(len(diffs), x_pos[i]), diffs,
                               color='black', alpha=0.25, s=20, zorder=2,
                               label="Individual diffs" if i == 0 else None)
            ax.set_title(f"{metric} ({METRICS_BETTER.get(metric, 'N/A')} = better)", fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups, rotation=40, ha='right', fontsize=8)
            ax.set_ylabel("Mean Difference", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
            if idx == 0:
                ax.legend(fontsize=8)
        # Hide any unused subplots
        for empty_idx in range(n, rows * cols):
            r, c = divmod(empty_idx, cols)
            axes[r][c].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ci_all_metrics.png"))
        plt.close(fig)

def compare_fedtabdiff(model_type, it):
    runs_df = get_avg_agents_df(it, model_type)
    runs_dir = get_runs_path(model_type)
    runs_results_dir = os.path.join(runs_dir, "results")
    os.makedirs(runs_results_dir, exist_ok=True)

    # Compare each group of agents in runs df with equivalent in FedTabDiff
    for n in [4, 7, 10]:
        baseline_df = get_fedtabdiff_df(n, dataset_name="adult")
        if baseline_df is None or baseline_df.empty:
            print(f"No FedTabDiff data available for {n} agents.")
            continue
        # Filter runs_df for n agents
        runs_n_df = runs_df[runs_df['run'].str.contains(f'{n}Agents')]
        if runs_n_df.empty:
            print(f"No runs data available for {n} agents.")
            continue

        methods = runs_n_df[runs_n_df['run'] != 'baseline_ctgan']
        numeric_cols = runs_n_df.select_dtypes(include='number').columns
        group_diffs = {}
        for method in methods['run'].unique():
            group_id = extract_group(method)
            method_data = methods[methods['run'] == method]
            method_diffs = {}
            for col in numeric_cols:
                if col in baseline_df.columns and col in method_data.columns:
                    diff = method_data[col].values - baseline_df[col].values
                    method_diffs.setdefault(col, []).append(diff)
            if group_id not in group_diffs:
                group_diffs[group_id] = {}
            for col, diff_list in method_diffs.items():
                group_diffs[group_id].setdefault(col, []).extend(diff_list)

        results = []
        for group_id, metrics in group_diffs.items():
            for col, diffs_list in metrics.items():
                all_diffs = np.concatenate(diffs_list)
                # Paired t-test CI
                mean_t, lower_t, upper_t, p_value_t = paired_t_ci(all_diffs)
                # Permutation CI
                results.append({
                    "Group": group_id,
                    "Metric": col,
                    "Mean_t": mean_t,
                    "CI_Lower_t": lower_t,
                    "CI_Upper_t": upper_t,
                    "p_value_t": p_value_t,
                })
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(runs_results_dir, f"{model_type}-{it}-fedtabdiff-ci.csv"))
        run_dir = get_runs_path(model_type)
        plots_dir = os.path.join(run_dir, "plots", "fedtabdiff", f"{n}A")
        os.makedirs(plots_dir, exist_ok=True)
        for metric in df["Metric"].unique():
            if metric == 'AdversarialAccuracy':
                print(metric)
            # subset = df[df["Metric"] == metric].sort_values("Mean_t", ascending=False)
            subset = df[df["Metric"] == metric].sort_values("Group", ascending=False)
            if subset["Mean_t"].notna().sum() == 0:
                continue
            subset = subset.dropna(subset=["Group", "Mean_t", "CI_Lower_t", "CI_Upper_t"])
            subset["Group"] = subset["Group"].astype(str)
            colors_t = np.where(subset["p_value_t"] < 0.05, "red", "tab:blue")
            plt.figure(figsize=(14, 7))
            # Paired t-test CI (gray)
            plt.errorbar(
                subset["Group"], subset["Mean_t"],
                yerr=[subset["Mean_t"] - subset["CI_Lower_t"], subset["CI_Upper_t"] - subset["Mean_t"]],
                fmt='o', capsize=7, markersize=8, color='tab:gray', ecolor='tab:gray', label="t-test CI"
            )
            plt.scatter(
                subset["Group"], subset["Mean_t"],
                c=colors_t, s=80, zorder=3, label="t-test Significant (p<0.05)"
            )
            for i, group in enumerate(subset["Group"]):
                # Get all individual differences for this group/metric
                diffs = np.concatenate(group_diffs[group][metric])
                plt.scatter([group] * len(diffs), diffs, color='black', alpha=0.3, s=30, zorder=2,
                            label="Individual diffs" if i == 0 else None)
            plt.title(
                f"{metric}: Mean Diff with 95% CI (t-test) ({METRICS_BETTER[metric] if metric in METRICS_BETTER else 'N/A'} = better)",
                fontsize=18)
            plt.ylabel("Mean Difference", fontsize=14)
            plt.xlabel("Group", fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{run_dir}/plots/ci_{metric}_t.png")
            plt.close()

        metrics = [m for m in df["Metric"].unique() if df[df["Metric"] == m]["Mean_t"].notna().sum() > 0]
        if metrics:
            import math
            n = len(metrics)
            cols = min(4, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
            for idx, metric in enumerate(metrics):
                r, c = divmod(idx, cols)
                ax = axes[r][c]
                # subset = df[df["Metric"] == metric].sort_values("Mean_t", ascending=False)
                subset = df[df["Metric"] == metric].sort_values("Group", ascending=False)
                subset = subset.dropna(subset=["Group", "Mean_t", "CI_Lower_t", "CI_Upper_t"])
                if subset.empty:
                    ax.axis('off')
                    continue
                colors_t = np.where(subset["p_value_t"] < 0.05, "red", "tab:blue")
                groups = subset["Group"].astype(str).tolist()
                x_pos = np.arange(len(groups))
                # Errorbars (t-test CI)
                y = subset["Mean_t"].values
                yerr = np.vstack([y - subset["CI_Lower_t"].values, subset["CI_Upper_t"].values - y])
                ax.errorbar(x_pos, y, yerr=yerr, fmt='o', capsize=5, markersize=6,
                            color='tab:gray', ecolor='tab:gray', label="t-test CI")
                ax.scatter(x_pos, y, c=colors_t, s=60, zorder=3, label="t-test Significant (p<0.05)")
                # Individual diffs without jitter (aligned to group x position)
                for i, group in enumerate(groups):
                    if group in group_diffs and metric in group_diffs[group]:
                        diffs = np.concatenate(group_diffs[group][metric])
                        ax.scatter(np.full(len(diffs), x_pos[i]), diffs,
                                   color='black', alpha=0.25, s=20, zorder=2,
                                   label="Individual diffs" if i == 0 else None)
                ax.set_title(f"{metric} ({METRICS_BETTER.get(metric, 'N/A')} = better)", fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups, rotation=40, ha='right', fontsize=8)
                ax.set_ylabel("Mean Difference", fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.5)
                if idx == 0:
                    ax.legend(fontsize=8)
            # Hide any unused subplots
            for empty_idx in range(n, rows * cols):
                r, c = divmod(empty_idx, cols)
                axes[r][c].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "ci_all_metrics.png"))
            plt.close(fig)
dataset_name = "churn"
compare("tabddpm", 1000, dataset_name=dataset_name)
compare("ctgan", 300, dataset_name=dataset_name)
#compare_fedtabdiff("tabddpm", 1000)