import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import permutation_test

from DeConSyn.evaluation.plot import get_avg_agents_df, get_baseline_df, get_runs_path


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


def permutation_ci(diff, n_resamples=10000, alpha=0.05, random_state=42):
    diff = np.asarray(diff, dtype=float)
    mean = np.mean(diff)
    # Bootstrapped CI
    rng = np.random.default_rng(random_state)
    boot_means = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_resamples)]
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    # Permutation p-value
    result = permutation_test(
        (np.zeros_like(diff), diff),
        statistic=lambda x, y: np.mean(y - x),
        permutation_type='samples',
        alternative='two-sided',
        n_resamples=n_resamples,
        random_state=random_state
    )
    p_value = result.pvalue
    return mean, ci_lower, ci_upper, p_value

METRICS = ['DCR', 'NNDR', 'AdversarialAccuracy', 'JS', 'KS', 'RepU', 'DiSCO', 'Mean', 'Median', 'Var']
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
    'KS': 'Higher'
}

def rank_models(df):
    privacy_metrics = ['DCR', 'NNDR', 'AdversarialAccuracy', 'RepU', 'DiSCO']
    df_priv = df[df['Metric'].isin(privacy_metrics)]
    pivot = df_priv.pivot(index="Group", columns="Metric", values="Mean_t")
    for metric, better in METRICS_BETTER.items():
        if better == 'lower' and metric in pivot.columns:
            pivot[metric] = -pivot[metric]
    normed = (pivot - pivot.min()) / (pivot.max() - pivot.min())
    normed['NormalizedScore'] = normed.sum(axis=1)
    ranking = normed["NormalizedScore"].sort_values(ascending=False)
    return ranking


it = 1000
model_type = 'tabddpm'
runs_avg = get_avg_agents_df(it, model_type)
numeric_cols = runs_avg.select_dtypes(include='number').columns
if runs_avg is None or runs_avg.empty:
    print("No data available for analysis.")
    raise SystemExit

# Remove run == run-20251020-154840-7Agents-30Epochs-1000Iterations-ring-tabddpm
runs_avg = runs_avg[runs_avg['run'] != 'run-20251020-154840-7Agents-30Epochs-1000Iterations-ring-tabddpm'].copy()

# Compute differences between each method and the baseline
baseline = get_baseline_df(model_type)
methods = runs_avg[runs_avg['run'] != 'baseline_ctgan']
group_diffs = {}
for method in methods['run'].unique():
    group_id = extract_group(method)
    method_data = methods[methods['run'] == method]
    method_diffs = {}
    for col in numeric_cols:
        if col in baseline.columns and col in method_data.columns:
            diff = method_data[col].values - baseline[col].values
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
run_dir = get_runs_path(model_type)
plots_dir = os.path.join(run_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
for metric in df["Metric"].unique():
    if metric == 'AdversarialAccuracy':
        print(metric)
    subset = df[df["Metric"] == metric].sort_values("Mean_t", ascending=False)
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
        # Jitter for visibility
        plt.scatter([group] * len(diffs), diffs, color='black', alpha=0.3, s=30, zorder=2,
                    label="Individual diffs" if i == 0 else None)
    plt.title(f"{metric}: Mean Diff with 95% CI (t-test) ({METRICS_BETTER[metric] if metric in METRICS_BETTER else 'N/A'} = better)", fontsize=18)
    plt.ylabel("Mean Difference", fontsize=14)
    plt.xlabel("Group", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/plots/ci_{metric}_t.png")
    plt.close()

ranking = rank_models(df)
plt.figure(figsize=(12, 6))
ranking.plot(kind='bar', color='skyblue')
plt.ylabel("Composite Privacy Score", fontsize=14)
plt.xlabel("Group/Model", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.title(
    "Composite Privacy Scores by Group: Sum of Normalized Mean Differences (Higher = Better)",
    fontsize=16
)
plt.tight_layout()
plt.savefig(f"{run_dir}/plots/normalized_privacy_ranking.png")
plt.close()

