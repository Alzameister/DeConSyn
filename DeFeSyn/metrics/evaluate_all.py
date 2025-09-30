import argparse
import gc
import glob
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
import torch

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.utils.io import load_model_pickle
from FEST.privacy_utility_framework.build.lib.privacy_utility_framework.metrics.privacy_metrics.privacy_metric_manager import \
    PrivacyMetricManager
from FEST.privacy_utility_framework.build.lib.privacy_utility_framework.metrics.utility_metrics.utility_metric_manager import \
    UtilityMetricManager
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks.inference_class import \
    InferenceCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks.linkability_class import \
    LinkabilityCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks.singlingout_class import \
    SinglingOutCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.adversarial_accuracy_class import \
    AdversarialAccuracyCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.dcr_class import \
    DCRCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.disco import \
    DisclosureCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.nndr_class import \
    NNDRCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.basic_stats import \
    BasicStatsCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.correlation import \
    CorrelationMethod, CorrelationCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.js_similarity import \
    JSCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import \
    KSCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.wasserstein import \
    WassersteinCalculator, WassersteinMethod


# ======================================================================
# Helpers
# ======================================================================
ITER_MODEL_RE = re.compile(r"iter-(\d+)-model\.pkl$")
ITER_WEIGHTS_RE = re.compile(r"iter-(\d+)-weights\.pt$")
DEFAULT_METRICS = [
    "DCR", "NNDR", "AdversarialAccuracy",
    "Disclosure",
    "BasicStats", "JS", "KS",
    "CorrelationPearson", "CorrelationSpearman", "PCA",
    # "PCA3D",      # enable if you want HTML
    # "Consensus",  # needs --run-dir
]

def csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_txt(path: Path, content: float | dict | str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(content, dict):
            for k, v in content.items():
                f.write(f"{k}: {v}\n")
        else:
            f.write(str(content))

def load_vectors_from_state_dict(path: Path) -> np.ndarray:
    sd = torch.load(path, map_location="cpu")

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from walk(x)
        else:
            try:
                arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                if np.issubdtype(arr.dtype, np.number) and arr.size:
                    yield arr.ravel()
            except Exception:
                return

    parts = [p for p in walk(sd)]
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)

def make_context(
    manifest_path: str,
    model_path: str,
    output_dir: str,
    original_name: str,
    synthetic_name: str,
    seed: int,
) -> dict:
    original = DatasetLoader(manifest_path).get_train()
    model = load_model_pickle(Path(model_path))
    synthetic = model.sample(len(original), seed)

    out = Path(output_dir)
    return {
        "original": original,
        "synthetic": synthetic,
        "original_name": original_name,
        "synthetic_name": synthetic_name,
        "out_priv": ensure_dir(out / "Privacy"),
        "out_util": ensure_dir(out / "Similarity"),
        "out_root": ensure_dir(out),
        "seed": seed,
    }

# ======================================================================
# Metrics
# ======================================================================

def privacy_eval(calc) -> float:
    m = PrivacyMetricManager()
    m.add_metric(calc)
    res = m.evaluate_all()
    return float(next(iter(res.values()))) if isinstance(res, dict) else float(res)

def utility_eval(calc) -> float:
    m = UtilityMetricManager()
    m.add_metric(calc)
    res = m.evaluate_all()
    return float(next(iter(res.values()))) if isinstance(res, dict) else float(res)


def run_dcr(ctx):  # -> (value, artifacts)
    # Dont run if it already exists
    if (ctx["out_priv"] / "DCR.txt").exists():
        with open(ctx["out_priv"] / "DCR.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    v = privacy_eval(DCRCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"]))
    save_txt(ctx["out_priv"] / "DCR.txt", {"DCR": v})
    return v, {}

def run_nndr(ctx):
    if (ctx["out_priv"] / "NNDR.txt").exists():
        with open(ctx["out_priv"] / "NNDR.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    v = privacy_eval(NNDRCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"]))
    save_txt(ctx["out_priv"] / "NNDR.txt", {"NNDR": v})
    return v, {}

def run_advacc(ctx):
    if (ctx["out_priv"] / "AdversarialAccuracy.txt").exists():
        with open(ctx["out_priv"] / "AdversarialAccuracy.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    v = privacy_eval(AdversarialAccuracyCalculator(
        original=ctx["original"], synthetic=ctx["synthetic"], original_name=["original_name"], synthetic_name=["synthetic_name"], distance_metric='euclidean'))
    save_txt(ctx["out_priv"] / "AdversarialAccuracy.txt", {"AdversarialAccuracy": v})
    return v, {}

def run_singlingout(ctx):
    if (ctx["out_priv"] / "SinglingOut.txt").exists():
        with open(ctx["out_priv"] / "SinglingOut.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    r = SinglingOutCalculator(original=ctx["original"], synthetic=ctx["synthetic"], original_name=["original_name"], synthetic_name=["synthetic_name"]).evaluate()
    v = float(getattr(r, "value", r))
    save_txt(ctx["out_priv"] / "SinglingOut.txt", {"SinglingOut": v})
    return v, {}

def run_inference(ctx, inf_aux_cols: List[str], secret: str, regression: bool):
    if (ctx["out_priv"] / "Inference.txt").exists():
        with open(ctx["out_priv"] / "Inference.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    r = InferenceCalculator(
        ctx["original"], ctx["synthetic"],
        aux_cols=inf_aux_cols, secret=secret, regression=regression,
        original_name=ctx["original_name"], synthetic_name=ctx["synthetic_name"],
    ).evaluate()
    v = float(getattr(r, "value", r))
    save_txt(ctx["out_priv"] / "Inference.txt", {"Inference": v})
    return v, {}

def run_linkability(ctx, link_aux_cols: Tuple[List[str], List[str]], control_frac: float):
    if (ctx["out_priv"] / "Linkability.txt").exists():
        with open(ctx["out_priv"] / "Linkability.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    ctrl = ctx["original"].sample(frac=control_frac, random_state=ctx["seed"]).reset_index(drop=True)
    base = ctx["original"].drop(ctrl.index).reset_index(drop=True)
    r = LinkabilityCalculator(
        original=base, synthetic=ctx["synthetic"], aux_cols=link_aux_cols, control=ctrl,
        original_name=ctx["original_name"], synthetic_name=ctx["synthetic_name"],
    ).evaluate()
    v = float(getattr(r, "value", r))
    save_txt(ctx["out_priv"] / "Linkability.txt", {"Linkability": v})
    return v, {}

def run_disclosure(ctx, keys: List[str], target: str):
    if (ctx["out_priv"] / "RepU.txt").exists() and (ctx["out_priv"] / "DiSCO.txt").exists():
        with open(ctx["out_priv"] / "RepU.txt", "r", encoding="utf-8") as f:
            repu = float(f.read().strip().split(":")[-1])
        with open(ctx["out_priv"] / "DiSCO.txt", "r", encoding="utf-8") as f:
            disco = float(f.read().strip().split(":")[-1])
        return {"RepU": repu, "DiSCO": disco}, {}
    repu, disco = DisclosureCalculator(
        ctx["original"], ctx["synthetic"], keys=keys, target=target,
        original_name=ctx["original_name"], synthetic_name=ctx["synthetic_name"],
    ).evaluate()
    save_txt(ctx["out_priv"] / "RepU.txt", {"RepU": repu})
    save_txt(ctx["out_priv"] / "DiSCO.txt", {"DiSCO": disco})
    return {"RepU": repu, "DiSCO": disco}, {}

def run_basicstats(ctx):
    if (ctx["out_util"] / "BasicStats.txt").exists():
        with open(ctx["out_util"] / "BasicStats.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            r = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        r[k.strip()] = float(v.strip())
                    except ValueError:
                        r[k.strip()] = v.strip()
        return r, {}
    r = BasicStatsCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"]).evaluate()
    save_txt(ctx["out_util"] / "BasicStats.txt", r)
    return r, {}

def run_js(ctx):
    if (ctx["out_util"] / "JS.txt").exists():
        with open(ctx["out_util"] / "JS.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    v = utility_eval(JSCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"]))
    save_txt(ctx["out_util"] / "JS.txt", {"JS": v})
    return v, {}

def run_ks(ctx):
    if (ctx["out_util"] / "KS.txt").exists():
        with open(ctx["out_util"] / "KS.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    v = utility_eval(KSCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"]))
    save_txt(ctx["out_util"] / "KS.txt", {"KS": v})
    return v, {}

def run_wasserstein(ctx):
    if (ctx["out_util"] / "WASSERSTEIN.txt").exists():
        with open(ctx["out_util"] / "WASSERSTEIN.txt", "r", encoding="utf-8") as f:
            v = float(f.read().strip().split(":")[-1])
        return v, {}
    wc = WassersteinCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"])
    v = wc.evaluate(metric=WassersteinMethod.SINKHORN, n_iterations=5, n_samples=500)
    save_txt(ctx["out_util"] / "WASSERSTEIN.txt", {"WASSERSTEIN": v})
    return v, {}

PRIVACY = {
    "DCR": run_dcr,
    "NNDR": run_nndr,
    "AdversarialAccuracy": run_advacc,
    "SinglingOut": run_singlingout,
    "Inference": run_inference,
    "Linkability": run_linkability,
    "Disclosure": run_disclosure,
}
UTILITY = {
    "BasicStats": run_basicstats,
    "JS": run_js,
    "KS": run_ks,
    "WASSERSTEIN": run_wasserstein,
}

# ======================================================================
# PCA and Correlation
# ======================================================================

def plot_correlation(ctx, method: CorrelationMethod, label: str) -> dict[str, str]:
    if (ctx["out_util"] / f"{label}_result.txt").exists():
        return {
            f"{label}_score_txt": str(ctx["out_util"] / f"{label}_result.txt"),
            f"{label}_orig_csv": str(ctx["out_util"] / f"{label}_original_correlation.csv"),
            f"{label}_syn_csv": str(ctx["out_util"] / f"{label}_synthetic_correlation.csv"),
            f"{label}_orig_heatmap": str(ctx["out_util"] / f"{label}_original_heatmap.png"),
            f"{label}_syn_heatmap": str(ctx["out_util"] / f"{label}_synthetic_heatmap.png"),
        }

    calc = CorrelationCalculator(ctx["original"], ctx["synthetic"], ctx["original_name"], ctx["synthetic_name"])
    score = calc.evaluate(method=method)
    save_txt(ctx["out_util"] / f"{label}_result.txt", {label: score})

    orig_df, syn_df = calc.correlation_pairs(method=method)
    p_orig_csv = ctx["out_util"] / f"{label}_original_correlation.csv"
    p_syn_csv = ctx["out_util"] / f"{label}_synthetic_correlation.csv"
    orig_df.to_csv(p_orig_csv, index=False)
    syn_df.to_csv(p_syn_csv, index=False)

    plt.figure(figsize=(9, 7))
    sns.heatmap(orig_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Original ({method.name.title()})")
    p1 = ctx["out_util"] / f"{label}_original_heatmap.png"
    plt.savefig(p1); plt.clf()

    plt.figure(figsize=(9, 7))
    sns.heatmap(syn_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Synthetic ({method.name.title()})")
    p2 = ctx["out_util"] / f"{label}_synthetic_heatmap.png"
    plt.savefig(p2); plt.clf()
    plt.close()

    return {
        f"{label}_score_txt": str(ctx["out_util"] / f"{label}_result.txt"),
        f"{label}_orig_csv": str(p_orig_csv),
        f"{label}_syn_csv": str(p_syn_csv),
        f"{label}_orig_heatmap": str(p1),
        f"{label}_syn_heatmap": str(p2),
    }


def plot_pca(ctx) -> str:
    combined = pd.concat([ctx["original"], ctx["synthetic"]], ignore_index=True)
    X = combined.select_dtypes(include=[np.number]).fillna(0)
    coords = PCA(n_components=2).fit_transform(X)
    n = len(ctx["original"])

    plt.figure(figsize=(9, 7))
    plt.scatter(coords[:n, 0], coords[:n, 1], label="Original", alpha=0.5)
    plt.scatter(coords[n:, 0], coords[n:, 1], label="Synthetic", alpha=0.5)
    plt.title("PCA: Original vs Synthetic"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    out = ctx["out_util"] / "PCA_Original_vs_Synthetic.png"
    plt.savefig(out); plt.clf()
    plt.close()
    return str(out)


def plot_pca3d(ctx) -> str:
    from sklearn.decomposition import PCA
    import plotly.graph_objs as go
    from plotly.offline import plot as plotly_plot

    combined = pd.concat([ctx["original"], ctx["synthetic"]], ignore_index=True)
    X = combined.select_dtypes(include=[np.number]).fillna(0)

    pca = PCA(n_components=3).fit(X)
    coords = pca.transform(X)
    n = len(ctx["original"])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=coords[:n, 0], y=coords[:n, 1], z=coords[:n, 2], mode="markers", name="Original", opacity=0.6, marker=dict(size=3)))
    fig.add_trace(go.Scatter3d(x=coords[n:, 0], y=coords[n:, 1], z=coords[n:, 2], mode="markers", name="Synthetic", opacity=0.6, marker=dict(size=3)))
    evr = pca.explained_variance_ratio_
    fig.update_layout(title=f"PCA 3D (PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%})",
                      scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    out = ctx["out_util"] / "PCA3D_Original_vs_Synthetic.html"
    plotly_plot(fig, filename=str(out), auto_open=False, include_plotlyjs="cdn")
    return str(out)


# ======================================================================
# Consensus
# ======================================================================
def collect_consensus(run_dir: Path) -> Tuple[List[str], List[int], dict[str, List[np.ndarray]]]:
    files = sorted(glob.glob(str(run_dir / "agent_*" / "iter-*-weights.pt")))
    per_agent: dict[str, dict[int, Path]] = {}
    for p in files:
        pth = Path(p)
        ag = pth.parent.name
        m = ITER_WEIGHTS_RE.search(pth.name)
        if not m:
            continue
        t = int(m.group(1))
        per_agent.setdefault(ag, {})[t] = pth
    if not per_agent:
        raise ValueError(f"No agent checkpoints under {run_dir}")

    common = sorted(set.intersection(*(set(d.keys()) for d in per_agent.values())))
    if not common:
        raise ValueError("Agents do not share common iterations.")

    agents = sorted(per_agent.keys())
    vecs = {ag: [load_vectors_from_state_dict(per_agent[ag][t]) for t in common] for ag in agents}
    return agents, common, vecs

def consensus_artifacts(out_dir: Path, run_dir: Path) -> dict[str, str]:
    ensure_dir(out_dir)
    agents, iters, vecs = collect_consensus(run_dir)
    iters_np = np.asarray(iters)
    eps = 1e-12

    E_abs = {ag: [] for ag in agents}
    E_rel = {ag: [] for ag in agents}
    D_abs, D_rel, V_abs, V_rel = [], [], [], []

    for k, _ in enumerate(iters):
        X = np.stack([vecs[ag][k] for ag in agents], axis=0)
        mu = X.mean(axis=0)
        mu_norm = max(float(np.linalg.norm(mu)), eps)
        diffs = X - mu

        for i, ag in enumerate(agents):
            e = float(np.linalg.norm(diffs[i]))
            E_abs[ag].append(e)
            E_rel[ag].append(e / mu_norm)

        # Diameter
        dmax = 0.0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                d = float(np.linalg.norm(X[i] - X[j]))
                if d > dmax: d = dmax if False else d  # keep readable
                dmax = max(dmax, d)
        D_abs.append(dmax)
        D_rel.append(dmax / mu_norm)

        v = float(np.sum(diffs ** 2))
        V_abs.append(v)
        V_rel.append(v / (len(agents) * max(mu_norm**2, eps**2)))

    # CSVs
    e_rel_df = pd.DataFrame({"iter": iters_np}); e_abs_df = pd.DataFrame({"iter": iters_np})
    for ag in agents:
        e_rel_df[ag] = E_rel[ag]; e_abs_df[ag] = E_abs[ag]
    p_e_rel = out_dir / "E_rel_per_agent.csv"; p_e_abs = out_dir / "E_abs_per_agent.csv"
    e_rel_df.to_csv(p_e_rel, index=False); e_abs_df.to_csv(p_e_abs, index=False)

    summary = pd.DataFrame({
        "iter": iters_np,
        "E_rel_mean": e_rel_df[agents].mean(axis=1),
        "E_rel_rms": np.sqrt((e_rel_df[agents].to_numpy()**2).mean(axis=1)),
        "E_abs_mean": e_abs_df[agents].mean(axis=1),
        "E_abs_rms": np.sqrt((e_abs_df[agents].to_numpy()**2).mean(axis=1)),
        "D_rel": D_rel, "D_abs": D_abs, "V_rel": V_rel, "V_abs": V_abs,
    })
    p_sum = out_dir / "consensus_summary.csv"; summary.to_csv(p_sum, index=False)

    # Plots
    def lineplot(xs, ys_dict, title, ylabel, outname, logy=True, with_legend=True):
        plt.figure()
        for k, ys in ys_dict.items():
            plt.plot(xs, ys, label=k)
        if logy: plt.yscale("log")
        plt.xlabel("Round"); plt.ylabel(ylabel); plt.title(title)
        if with_legend: plt.legend(fontsize="small", ncol=2)
        plt.tight_layout(); path = out_dir / outname
        plt.savefig(path); plt.clf()
        plt.close()
        return str(path)

    per_agent_plot = lineplot(iters_np, E_rel, f"Consensus error (relative) — {run_dir.name}",
                              "Relative consensus error", "consensus-error-per-agent.png")

    plt.figure()
    plt.plot(iters_np, summary["E_rel_mean"], label="Mean (rel)", linewidth=2)
    plt.plot(iters_np, summary["E_abs_mean"], label="Mean (abs)", linewidth=2)
    plt.yscale("log"); plt.xlabel("Round"); plt.ylabel("Consensus error"); plt.title(f"Average consensus — {run_dir.name}")
    plt.legend(); plt.tight_layout()
    avg_plot = out_dir / "consensus-error-average.png"; plt.savefig(avg_plot); plt.clf()
    plt.close()

    def dual(y1, y2, title, ylabel, outname):
        plt.figure()
        plt.plot(iters_np, y1, label="Normalized", linewidth=2)
        plt.plot(iters_np, y2, label="Absolute", linestyle="--", linewidth=2)
        plt.yscale("log"); plt.xlabel("Round"); plt.ylabel(ylabel); plt.title(title)
        plt.legend(); plt.tight_layout()
        out = out_dir / outname; plt.savefig(out); plt.clf();
        plt.close()
        return str(out)

    diameter_plot = dual(summary["D_rel"], summary["D_abs"], f"Network diameter — {run_dir.name}",
                         "Diameter (max pairwise dist)", "consensus-diameter.png")
    potential_plot = dual(summary["V_rel"], summary["V_abs"], f"Consensus potential — {run_dir.name}",
                          "Potential Σ||θ_i−μ||²", "consensus-potential.png")

    return {
        "E_rel_per_agent_csv": str(p_e_rel),
        "E_abs_per_agent_csv": str(p_e_abs),
        "summary_csv": str(p_sum),
        "per_agent_plot": per_agent_plot,
        "avg_error_plot": str(avg_plot),
        "diameter_plot": diameter_plot,
        "potential_plot": potential_plot,
    }


# ======================================================================
# Core
# ======================================================================
def evaluate(
        manifest_path: str,
        run_path: str,
        model_path: str,
        output_path: str,
        original_name: str,
        synthetic_name: str,
        seed: int,
        metrics: List[str],
        # metric params
        keys: List[str] = [],
        target: str = "",
        inf_aux_cols: List[str] = [],
        secret: str = "",
        regression: bool = False,
        link_a: List[str] = [],
        link_b: List[str] = [],
        control_frac: float = 0.3
) -> dict[str, dict]:
    ctx = make_context(manifest_path, model_path, output_path, original_name, synthetic_name, seed)
    privacy_out, utility_out, artifacts = {}, {}, {}
    results = {}

    # Privacy
    for m in metrics:
        if m not in PRIVACY: continue
        if m == "Disclosure":
            if not keys or not target: raise ValueError("Disclosure needs --keys and --target.")
            val, art = run_disclosure(ctx, keys, target)
        elif m == "Inference":
            if not inf_aux_cols or not secret: raise ValueError("Inference needs --inf-aux-cols and --secret.")
            val, art = run_inference(ctx, inf_aux_cols, secret, regression)
        elif m == "Linkability":
            if not link_a or not link_b: raise ValueError("Linkability needs --link-a and --link-b.")
            val, art = run_linkability(ctx, (link_a, link_b), control_frac)
        else:
            val, art = PRIVACY[m](ctx)

        privacy_out[m] = val
        artifacts.update(art)
        try:
            results[m] = float(getattr(val, "value", val))
        except Exception:
            results[m] = str(val)

    # Utility
    for m in metrics:
        if m not in UTILITY: continue
        val, art = UTILITY[m](ctx)
        utility_out[m] = val;
        artifacts.update(art)
        try:
            results[m] = float(getattr(val, "value", val))
        except Exception:
            results[m] = str(val)

    # PCA and Correlation
    if "CorrelationPearson" in metrics:
        artifacts.update(plot_correlation(ctx, CorrelationMethod.PEARSON, "CorrelationPearson"))
    if "CorrelationSpearman" in metrics:
        artifacts.update(plot_correlation(ctx, CorrelationMethod.SPEARMAN, "CorrelationSpearman"))
    if "PCA" in metrics:
        artifacts["PCA"] = plot_pca(ctx)
    if "PCA3D" in metrics:
        artifacts["PCA3D"] = plot_pca3d(ctx)
    if "Consensus" in metrics:
        if not run_path: raise ValueError("Consensus needs --run-dir.")
        artifacts.update(consensus_artifacts(ensure_dir(Path(output_path) / "Consensus"), run_path))

    results_csv = Path(output_path) / "results.csv"
    df_row = pd.DataFrame([results], index=[synthetic_name])
    if results_csv.exists():
        df = pd.read_csv(results_csv, index_col=0)
        df.update(df_row)
        df = df.combine_first(df_row)
    else:
        df = df_row
    df.to_csv(results_csv)
    return {"privacy": privacy_out, "utility": utility_out, "artifacts": artifacts}

def get_agents(run_path: Path):
    agents = sorted(glob.glob(str(run_path / "agent_*")))
    return [Path(a).name for a in agents if Path(a).is_dir()]

def evaluate_iterations(
        manifest_path: str,
        run_path: Path,
        output_path: str,
        original_name: str,
        synthetic_name: str,
        seed: int,
        metrics: List[str],
        iter_interval: int = 50,
        # metric params
        keys: List[str] = [],
        target: str = "",
        inf_aux_cols: List[str] = [],
        secret: str = "",
        regression: bool = False,
        link_a: List[str] = [],
        link_b: List[str] = [],
        control_frac: float = 0.3
):
    # Find agents
    agents = get_agents(run_path)
    if not agents:
        raise ValueError(f"No agents found in {run_path}")

    for agent in agents:
        agent_path = run_path / agent
        model_files = sorted(glob.glob(str(agent_path / "iter-*-model.pkl")))
        if not model_files:
            print(f"No model files found for {agent} in {agent_path}, skipping.")
            continue

        for model in model_files:
            m = Path(model)
            it_m = ITER_MODEL_RE.search(m.name)
            if not it_m or (int(it_m.group(1)) % iter_interval) != 0:
                print(f"Model file {m} does not match expected pattern, skipping.")
                continue
            it = int(it_m.group(1))
            out_dir = Path(output_path) / agent / f"iter-{it:04d}"
            print(f"Evaluating {agent} iteration {it}...")
            try:
                evaluate(
                    manifest_path=manifest_path,
                    run_path=str(run_path),
                    model_path=str(m),
                    output_path=str(out_dir),
                    original_name=original_name,
                    synthetic_name=f"{synthetic_name}-{agent}-iter{it:04d}",
                    seed=seed,
                    metrics=metrics,
                    keys=keys,
                    target=target,
                    inf_aux_cols=inf_aux_cols,
                    secret=secret,
                    regression=regression,
                    link_a=link_a,
                    link_b=link_b,
                    control_frac=control_frac
                )
            except Exception as e:
                print(f"Error evaluating {agent} iteration {it}: {e}")

# ======================================================================
# CLI
# ======================================================================
def build_parser():
    p = argparse.ArgumentParser(description="Evaluate privacy/utility metrics for synthetic tabular data.")
    # Required
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--model-path", required=True, help="Filename of pickled model (inside agent_* or as baseline path).")
    p.add_argument("--output-path", required=True)
    p.add_argument("--run-path", default=None, help="Parent folder with agent_* for per-agent or iteration eval.")
    # Labels
    p.add_argument("--original-name", default="adult")
    p.add_argument("--synthetic-name", default="CTGAN")
    # Modes
    p.add_argument("--baseline", action="store_true", help="Treat --model-name as a full path, not inside agent_*.")
    p.add_argument("--iter-eval", action="store_true")
    p.add_argument("--iter-interval", type=int, default=50)
    # Metrics
    p.add_argument("--metrics", type=csv_list, default=DEFAULT_METRICS)
    # Disclosure
    p.add_argument("--keys", type=csv_list, default=[])
    p.add_argument("--target", default=None)
    # Inference
    p.add_argument("--inf-aux-cols", type=csv_list, default=[])
    p.add_argument("--secret", default=None)
    p.add_argument("--regression", action="store_true")
    # Linkability
    p.add_argument("--link-a", type=csv_list, default=[])
    p.add_argument("--link-b", type=csv_list, default=[])
    p.add_argument("--control-frac", type=float, default=0.3)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    metrics = [m.strip() for m in args.metrics]

    link_a, link_b = args.link_a, args.link_b

    if args.baseline:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"ERROR: baseline model not found: {model_path}", file=sys.stderr)
            return 2

        evaluate(
            manifest_path=args.manifest_path,
            run_path=args.run_path,
            model_path=str(model_path),
            output_path=args.output_path,
            original_name=args.original_name,
            synthetic_name=args.synthetic_name,
            seed=args.seed,
            metrics=metrics,
            keys=args.keys,
            target=args.target or "",
            inf_aux_cols=args.inf_aux_cols,
            secret=args.secret or "",
            regression=args.regression,
            link_a=link_a,
            link_b=link_b,
            control_frac=args.control_frac
        )

    elif args.iter_eval:
        # Per agent evaluation
        agents = get_agents(Path(args.run_path))
        if not agents:
            print(f"ERROR: no agents found in --run-path: {args.run_path}", file=sys.stderr)
            return 2

        for agent in agents:
            run_path = Path(args.run_path) if args.run_path else None
            agent_dir = Path(args.run_path) / agent
            model_path = Path(args.run_path) / agent / args.model_path
            model_files = sorted(glob.glob(str(agent_dir / "iter-*-model.pkl")))
            if not model_files:
                print(f"WARNING: no model files found for {agent}, skipping: {agent_dir}", file=sys.stderr)
                continue

            for model_path in model_files:
                m = Path(model_path)
                match = ITER_MODEL_RE.search(m.name)
                if not match:
                    continue
                it = int(match.group(1))
                if it % 100 != 0:
                    continue
                out_dir = agent_dir / f"results_iter_{it:04d}"
                print(f"Evaluating {agent} at iteration {it}...")
                evaluate(
                    manifest_path=args.manifest_path,
                    run_path=str(run_path) if run_path else None,
                    model_path=str(m),
                    output_path=str(out_dir),
                    original_name=args.original_name,
                    synthetic_name=f"{args.synthetic_name}-{agent}-iter{it:04d}",
                    seed=args.seed,
                    metrics=metrics,
                    keys=args.keys,
                    target=args.target or "",
                    inf_aux_cols=args.inf_aux_cols,
                    secret=args.secret or "",
                    regression=args.regression,
                    link_a=link_a,
                    link_b=link_b,
                    control_frac=args.control_frac
                )
                gc.collect()

    else:
        # Per agent evaluation
        agents = get_agents(Path(args.run_path))
        if not agents:
            print(f"ERROR: no agents found in --run-path: {args.run_path}", file=sys.stderr)
            return 2

        for agent in agents:
            run_path = Path(args.run_path) if args.run_path else None
            model_path = Path(args.run_path) / agent / args.model_path
            if not model_path.exists():
                print(f"WARNING: model not found for {agent}, skipping: {model_path}", file=sys.stderr)
                continue
            out_dir = Path(args.run_path) / agent / "results_iter_0500"
            print(f"Evaluating {agent}...")

            evaluate(
                manifest_path=args.manifest_path,
                run_path=str(run_path) if run_path else None,
                model_path=str(model_path),
                output_path=str(out_dir),
                original_name=args.original_name,
                synthetic_name=args.synthetic_name,
                seed=args.seed,
                metrics=metrics,
                keys=args.keys,
                target=args.target or "",
                inf_aux_cols=args.inf_aux_cols,
                secret=args.secret or "",
                regression=args.regression,
                link_a=link_a,
                link_b=link_b,
                control_frac=args.control_frac
            )
            gc.collect()
    return 0

if __name__ == "__main__":
    # Get all run dirs and evaluate each seperately
    for run_dir in sorted(glob.glob("C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/*")):
        if not Path(run_dir).is_dir():
            continue
        # Check if path contains subdirs
        subdirs = [d for d in Path(run_dir).iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                print(f"Evaluating run dir: {subdir}")
                # Only evaluate if it contains agent_*
                agents = get_agents(subdir)
                if not agents:
                    print(f"WARNING: no agents found in run dir: {subdir}, skipping.", file=sys.stderr)
                    continue
                # Evaluate
                # Name of subdir is used as synthetic name
                subdir_name = subdir.name
                # Split up into parts '-'
                parts = subdir_name.split('-')
                # Get integers of each part (example: 10Agents --> 10)
                int_parts = [int(re.findall(r'\d+', part)[0]) for part in parts if re.findall(r'\d+', part)]
                int_parts = int_parts[1:]
                # Synthetic name = "CTGAN 7A 1E 500R SmallWorld"
                synthetic_name = "CTGAN " + str(int_parts[0]) + "A " + str(int_parts[1]) + "E " + str(int_parts[2]) + "R " + parts[-1]
                sys.argv = [
                    sys.argv[0],
                    "--manifest-path", "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult/manifest.yaml",
                    "--run-path", str(subdir),
                    "--model-path", "iter-00500-model.pkl",
                    "--output-path", str(subdir / "evaluation"),
                    "--original-name", "adult",
                    "--synthetic-name", "CTGAN",
                    "--metrics", ",".join(DEFAULT_METRICS),
                    "--seed", "42",
                    "--keys", "age,education,marital-status,occupation",
                    "--target", "income",
                    "--inf-aux-cols", "age,education,marital-status,occupation",
                    "--secret", "income",
                    "--link-a", "age,education,marital-status,occupation",
                    "--link-b", "race,sex,relationship",
                    "--control-frac", "0.3",
                ]
                main()

    raise SystemExit(main())



