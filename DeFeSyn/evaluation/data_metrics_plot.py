# This notebook cell defines a robust parser and plotting utilities for your two .txt metric files.
# 1) Upload your two .txt files to the left (Files) and rename them to:
#       /mnt/data/metrics_a.txt   and   /mnt/data/metrics_b.txt
# 2) Re-run this cell. If the files exist, it will parse and plot automatically.
#    The figures will also be saved as PNGs you can download.
#
# You can also call `parse_and_plot("/mnt/data/your1.txt", "/mnt/data/your2.txt")`
# at the bottom if you prefer custom paths.

import os
import re
import ast
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd

STAT_KEYS_ORDER = [
    "Pearson",
    "Spearman",
    "Mean",
    "Median",
    "Var",
]

PRIVACY_KEYS_ORDER = [
    "SinglingOut",  # value with optional CI
    "JS",
    "KS",
    "NNDR",
    "AdversarialAccuracy"
]

def parse_metrics_text(text: str) -> Dict[str, Any]:
    """
    Parse a single metrics text blob into a structured dict.
    Handles patterns like:
      - BasicStatsCalculator('adult', 'CTGAN'): {'mean': 0.0173, 'median': 0.00082, 'var': 0.01225}
      - CorrelationCalculator (Pearson): 0.9892
      - CorrelationCalculator (Spearman): 0.9889
      - AdversarialAccuracyCalculator('adult', 'CTGAN'): 0.8243
      - JSCalculator('adult', 'CTGAN'): 0.9614
      - KSCalculator('adult', 'CTGAN'): 0.9720
      - NNDRCalculator('adult', 'CTGAN'): 0.8979
      - SinglingOutCalculator('adult', 'Synthetic_Dataset'): PrivacyRisk(value=0.11, ci=(0.08, 0.14))
    Returns keys like "Mean", "Median", "Var", "Pearson", "Spearman", "AdversarialAccuracy",
    "JS", "KS", "NNDR", and "SinglingOut" (with sub-keys 'value', 'ci_low', 'ci_high' if found).
    """
    out: Dict[str, Any] = {}

    # Basic stats dict
    # Find the first {...} after BasicStatsCalculator:
    m_stats = re.search(r"BasicStatsCalculator\([^)]*\):\s*({.*})", text, flags=re.DOTALL)
    if m_stats:
        try:
            stats_dict = ast.literal_eval(m_stats.group(1))
            if isinstance(stats_dict, dict):
                if "mean" in stats_dict:
                    out["Mean"] = float(stats_dict["mean"])
                if "median" in stats_dict:
                    out["Median"] = float(stats_dict["median"])
                if "var" in stats_dict:
                    out["Var"] = float(stats_dict["var"])
        except Exception:
            pass

    # Correlations
    m_pearson = re.search(r"CorrelationCalculator\s*\(Pearson\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if m_pearson:
        out["Pearson"] = float(m_pearson.group(1))

    m_spearman = re.search(r"CorrelationCalculator\s*\(Spearman\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if m_spearman:
        out["Spearman"] = float(m_spearman.group(1))

    # AdversarialAccuracy
    m_adv = re.search(r"AdversarialAccuracyCalculator\([^)]*\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if m_adv:
        out["AdversarialAccuracy"] = float(m_adv.group(1))

    # JS / KS / NNDR
    m_js = re.search(r"JSCalculator\([^)]*\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if m_js:
        out["JS"] = float(m_js.group(1))

    m_ks = re.search(r"KSCalculator\([^)]*\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if m_ks:
        out["KS"] = float(m_ks.group(1))

    m_nndr = re.search(r"NNDRCalculator\([^)]*\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text)
    if m_nndr:
        out["NNDR"] = float(m_nndr.group(1))

    # SinglingOut: PrivacyRisk(value=..., ci=(low, high))
    m_so = re.search(
        r"SinglingOutCalculator\([^)]*\)\s*:\s*PrivacyRisk\(\s*value\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*ci\s*=\s*\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)\s*\)",
        text
    )
    if m_so:
        out["SinglingOut"] = {
            "value": float(m_so.group(1)),
            "ci_low": float(m_so.group(2)),
            "ci_high": float(m_so.group(3)),
        }
    else:
        # Fallback if we only have a value without CI (rare)
        m_so_val = re.search(
            r"SinglingOutCalculator\([^)]*\)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text
        )
        if m_so_val:
            out["SinglingOut"] = {"value": float(m_so_val.group(1))}

    return out

def read_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def metrics_to_dataframe(name_a: str, data_a: Dict[str, Any],
                         name_b: str, data_b: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (stat_df, privacy_df). privacy_df includes columns for CI if present.
    """
    # Statistical / utility metrics
    stat_rows = []
    for key in STAT_KEYS_ORDER:
        a_val = data_a.get(key, None)
        b_val = data_b.get(key, None)
        if a_val is None and b_val is None:
            continue
        if isinstance(a_val, dict) or isinstance(b_val, dict):
            # skip dict-type values here; basic stats are already extracted into keys above
            continue
        stat_rows.append({"metric": key, name_a: a_val, name_b: b_val})
    stat_df = pd.DataFrame(stat_rows)

    # Privacy metrics (currently: SinglingOut)
    priv_rows = []
    for key in PRIVACY_KEYS_ORDER:
        a_val = data_a.get(key, None)
        b_val = data_b.get(key, None)
        if a_val is None and b_val is None:
            continue
        if isinstance(a_val, dict) or isinstance(b_val, dict):
            # skip dict-type values here; basic stats are already extracted into keys above
            continue
        priv_rows.append({"metric": key, name_a: a_val, name_b: b_val})
    privacy_df = pd.DataFrame(priv_rows)
    return stat_df, privacy_df

def plot_statistical_bar(stat_df: pd.DataFrame, name_a: str, name_b: str, out_path: str = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/statistical_results.png"):
    if stat_df.empty:
        print("No statistical metrics to plot.")
        return
    # Build X and grouped bars
    x = range(len(stat_df))
    width = 0.4

    plt.figure()
    plt.bar([i - width/2 for i in x], stat_df[name_a], width=width, label=name_a)
    plt.bar([i + width/2 for i in x], stat_df[name_b], width=width, label=name_b)
    plt.xticks(list(x), stat_df["metric"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Statistical/Utility Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()
    print(f"Saved: {out_path}")

def plot_privacy_bar(privacy_df: pd.DataFrame, name_a: str, name_b: str, out_path: str = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/privacy_results.png"):
    if privacy_df.empty:
        print("No statistical metrics to plot.")
        return
    # Build X and grouped bars
    x = range(len(privacy_df))
    width = 0.4

    plt.figure()
    plt.bar([i - width / 2 for i in x], privacy_df[name_a], width=width, label=name_a)
    plt.bar([i + width / 2 for i in x], privacy_df[name_b], width=width, label=name_b)
    plt.xticks(list(x), privacy_df["metric"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Privacy Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()
    print(f"Saved: {out_path}")

def parse_and_plot(path_a: str, path_b: str, name_a: Optional[str] = None, name_b: Optional[str] = None):
    """
    High-level helper: parse two files and produce two figures:
        - statistical_metrics.png
        - privacy_metrics.png
    Also displays tidy dataframes for verification.
    """
    txt_a = read_file(path_a)
    txt_b = read_file(path_b)

    if txt_a is None:
        print(f"File not found: {path_a}")
        return
    if txt_b is None:
        print(f"File not found: {path_b}")
        return

    if name_a is None:
        name_a = os.path.basename(path_a)
    if name_b is None:
        name_b = os.path.basename(path_b)

    data_a = parse_metrics_text(txt_a)
    data_b = parse_metrics_text(txt_b)

    stat_df, privacy_df = metrics_to_dataframe(name_a, data_a, name_b, data_b)

    # Make plots
    plot_statistical_bar(stat_df, name_a, name_b)
    plot_privacy_bar(privacy_df, name_a, name_b)

# --- Auto-run if default file names exist ---
DEFAULT_A = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/run-20250831-171833-a4-e15-i20-alpha1.0-ring/Decentralized.txt"
DEFAULT_B = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/baseline_ctgan/Baseline.txt"
if os.path.exists(DEFAULT_A) and os.path.exists(DEFAULT_B):
    parse_and_plot(DEFAULT_A, DEFAULT_B)
else:
    print("Upload your files as /mnt/data/metrics_a.txt and /mnt/data/metrics_b.txt, then re-run this cell.")
    print("Or call parse_and_plot('/mnt/data/your1.txt','/mnt/data/your2.txt').")
