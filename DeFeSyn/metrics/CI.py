import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from DeFeSyn.metrics.plot import get_avg_agents_df, get_baseline_df, get_runs_path


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

it = 400
METRICS = ['DCR', 'NNDR', 'AdversarialAccuracy', 'JS', 'KS', 'RepU', 'DiSCO', 'Mean', 'Median', 'Var']
METRICS_BETTER = {
    'DCR': 'higher',
    'NNDR': 'higher',
    'AdversarialAccuracy': 'higher',
    'RepU': 'lower',
    'DiSCO': 'lower'
}
runs_avg = get_avg_agents_df(it)
numeric_cols = runs_avg.select_dtypes(include='number').columns
if runs_avg is None or runs_avg.empty:
    print("No data available for analysis.")
    raise SystemExit

# Compute differences between each method and the baseline
baseline = get_baseline_df()
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
        total_obs = sum(len(d) for d in diffs_list)
        # print(f"{group_id}, {col}: {total_obs}")
        all_diffs = np.concatenate(diffs_list)
        mean, lower, upper, p_value = paired_t_ci(all_diffs)
        results.append({
            "Group": group_id,
            "Metric": col,
            "Mean": mean,
            "CI_Lower": lower,
            "CI_Upper": upper,
            "p_value": p_value
        })
df = pd.DataFrame(results)
run_dir = get_runs_path()
for metric in df["Metric"].unique():

    subset = df[df["Metric"] == metric].sort_values("Mean", ascending=False)
    if subset["Mean"].notna().sum() == 0:
        continue
    subset = subset.dropna(subset=["Group", "Mean", "CI_Lower", "CI_Upper", "p_value"])
    subset["Group"] = subset["Group"].astype(str)
    colors = np.where(subset["p_value"] < 0.05, "red", "tab:blue")
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        subset["Group"], subset["Mean"],
        yerr=[subset["Mean"] - subset["CI_Lower"], subset["CI_Upper"] - subset["Mean"]],
        fmt='o', capsize=7, markersize=8, color='tab:gray', ecolor='tab:gray'
    )
    plt.scatter(
        subset["Group"], subset["Mean"],
        c=colors, s=80, zorder=3, label="Significant (p<0.05)"
    )
    plt.title(f"{metric}: Mean Diff with 95% CI", fontsize=18)
    plt.ylabel("Mean Difference", fontsize=14)
    plt.xlabel("Group", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/plots/ci_{metric}.png")