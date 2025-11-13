import glob
import itertools
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

ITER_RE = re.compile(r"iter-(\d+)-weights\.pt$")

def _is_tensor(x):
    return isinstance(x, torch.Tensor)

def _load_vec(path: Path) -> np.ndarray:
    """Load and flatten all numeric parameters from a file."""
    sd = torch.load(path, map_location="cpu")

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from walk(x)
        else:
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if hasattr(arr, "dtype") and np.issubdtype(arr.dtype, np.number):
                yield arr.ravel()

    parts = [p for p in walk(sd) if p.size > 0]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

def _max_pairwise_distance(X):
    return max(float(np.linalg.norm(a - b)) for a, b in itertools.combinations(X, 2))

def _plot_per_agent_metric(ax, iters, metric_dict, ylabel, title, yscale=None):
    """Helper to plot per-agent metrics."""
    for ag, vals in metric_dict.items():
        ax.plot(iters, vals, label=ag, alpha=0.9)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend(fontsize="small", ncol=2)

def extract_run_info(run_dir: Path) -> str:
    # Example: run-20251017-112942-10Agents-30Epochs-1000Iterations-full-tabddpm
    name = run_dir.name
    match = re.search(
        r'run-\d+-\d+-(\d+)Agents-(\d+)Epochs-(\d+)Iterations-([^-]+)-([^-]+)', name
    )
    if not match:
        raise ValueError("Run directory name does not match expected pattern.")
    agents, epochs, iters, topology, model_type = match.groups()
    return f"{agents}A {epochs}E {iters}R {topology.capitalize()} {model_type.capitalize()}"

def collect_iters(run_dir: Path):
    pattern = run_dir / "agent_*" / "iter-*-weights.pt"
    files = sorted(glob.glob(str(pattern)))
    by_agent = {}
    for p in files:
        pth = Path(p)
        ag = pth.parent.name
        m = ITER_RE.search(pth.name)
        if not m:
            continue
        t = int(m.group(1))
        by_agent.setdefault(ag, {})[t] = pth

    if not by_agent:
        raise ValueError(f"No agent checkpoints found under {run_dir}")

    # Align on common iterations across all agents
    common_iters = sorted(set.intersection(*(set(d.keys()) for d in by_agent.values())))
    if not common_iters:
        raise ValueError("Agents do not share common iterations (no intersection).")

    agents = sorted(by_agent.keys())
    vecs = {ag: [_load_vec(by_agent[ag][t]) for t in common_iters] for ag in agents}

    # Check equal parameter vector shapes across agents (optional strictness)
    ref_len = {len(v[0]) for v in vecs.values()}
    if len(ref_len) != 1:
        print("WARNING: Not all agents have identical parameter vector lengths.", file=sys.stderr)

    return agents, common_iters, vecs

def per_layer_means(param_dict):
    means = {}
    for k, v in param_dict.items():
        if isinstance(v, dict):
            means[k] = per_layer_means(v)
        else:
            if _is_tensor(v):
                means[k] = v.float().mean().item()
            elif isinstance(v, np.ndarray):
                means[k] = float(v.mean())
            else:
                try:
                    arr = np.asarray(v)
                    means[k] = float(arr.mean())
                except Exception:
                    means[k] = None
    return means

def _flatten_means(means, prefix=''):
    flat = {}
    for k, v in means.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_means(v, key))
        else:
            flat[key] = v
    return flat

def get_consensus_metrics(run_dir: Path, eps: float = 1e-12):
    """
    Compute consensus metrics (per-agent error E, max pairwise dist, sum of squared distance)
    in both absolute and normalized forms.

    Returns a dict with:
      agents, iters, E_abs/E_rel (dict of lists), D_abs/D_rel (lists), V_abs/V_rel (lists)
    """
    agents, iters, vecs = collect_iters(run_dir)

    E_abs = {ag: [] for ag in agents}
    E_rel = {ag: [] for ag in agents}
    D_abs, D_rel = [], []
    V_abs, V_rel = [], []

    n_agents = len(agents)

    for k, _t in enumerate(iters):
        X = np.stack([vecs[ag][k] for ag in agents], axis=0)
        mu = X.mean(axis=0)
        mu_norm = float(np.linalg.norm(mu))
        mu_norm_safe = max(mu_norm, eps)

        diffs = X - mu

        # Per-agent errors
        for i, ag in enumerate(agents):
            e_abs = float(np.linalg.norm(diffs[i]))
            E_abs[ag].append(e_abs)
            E_rel[ag].append(e_abs / mu_norm_safe)

        # Max pairwise distance
        dmax_abs = _max_pairwise_distance(X)
        D_abs.append(dmax_abs)
        D_rel.append(dmax_abs / mu_norm_safe)

        # Sum of squares
        v_abs = float(np.sum(diffs ** 2))
        V_abs.append(v_abs)
        denom_v = n_agents * max(mu_norm ** 2, eps ** 2)
        V_rel.append(v_abs / denom_v)

    return {
        "agents": agents,
        "iters": iters,
        "E_abs": E_abs,
        "E_rel": E_rel,
        "D_abs": D_abs,
        "D_rel": D_rel,
        "V_abs": V_abs,
        "V_rel": V_rel,
    }

def plot_per_layer_consensus(run_dir: Path, output_dir: Path):
    agent_dirs = sorted(run_dir.glob("agent_*"))
    per_layer = defaultdict(lambda: defaultdict(list))  # layer -> agent -> [means]
    iter_nums = None

    # Collect per-layer means
    for agent_dir in agent_dirs:
        agent = agent_dir.name
        model_files = sorted(agent_dir.glob("iter-*-weights.pt"))
        agent_iters = []
        for mf in model_files:
            m = ITER_RE.search(mf.name)
            if m:
                agent_iters.append((int(m.group(1)), mf))
        agent_iters.sort()
        if iter_nums is None:
            iter_nums = [t for t, _ in agent_iters]
        for t, mf in agent_iters:
            state = torch.load(mf, map_location="cpu")
            means = per_layer_means(state)
            flat_means = _flatten_means(means)
            for layer, mean in flat_means.items():
                per_layer[layer][agent].append(mean)

    step = 1
    if iter_nums is not None:
        idxs = np.arange(0, len(iter_nums), step)
        iter_nums_sub = [iter_nums[i] for i in idxs]
    else:
        iter_nums_sub = []

    # Plot all layers in one figure (one subplot per layer)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = list(per_layer.keys())
    n_layers = len(layers)
    ncols = 2
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        for agent, means in per_layer[layer].items():
            means_sub = [means[i] for i in idxs]
            ax.plot(iter_nums_sub, means_sub, label=agent)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean")
        ax.set_title(f"{layer}")
        ax.legend(fontsize="small")
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "consensus_per_layer_all_layers.png"
    plt.savefig(fig_path)
    plt.close(fig)

    return { "per_layer_plot": str(fig_path) }

def plot_total_consensus(res: dict, out_dir: Path, run_name: str) -> dict[str, str]:
    agents, iters = res["agents"], np.asarray(res["iters"])
    step = 1
    idxs = np.arange(0, len(iters), step)
    iters = iters[idxs]

    # Prepare dataframes as before...
    e_rel_df = pd.DataFrame({"iter": iters})
    e_abs_df = pd.DataFrame({"iter": iters})
    for ag in agents:
        e_rel_df[ag] = np.array(res["E_rel"][ag])[idxs]
        e_abs_df[ag] = np.array(res["E_abs"][ag])[idxs]
    Erel_mat = e_rel_df[agents].to_numpy()
    Eabs_mat = e_abs_df[agents].to_numpy()
    avg_df = pd.DataFrame({
        "iter": iters,
        "E_rel_mean": Erel_mat.mean(axis=1),
        "E_rel_rms": np.sqrt((Erel_mat ** 2).mean(axis=1)),
        "E_abs_mean": Eabs_mat.mean(axis=1),
        "E_abs_rms": np.sqrt((Eabs_mat ** 2).mean(axis=1)),
        "D_rel": np.array(res["D_rel"])[idxs],
        "D_abs": np.array(res["D_abs"])[idxs],
        "V_rel": np.array(res["V_rel"])[idxs],
        "V_abs": np.array(res["V_abs"])[idxs],
    })


    # --- Combined Figure ---
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    axes = axes.flatten()

    # 1. Per-agent relative error (log)
    _plot_per_agent_metric(axes[0], iters, {ag: e_rel_df[ag].values for ag in agents}, "Relative distance",
                           "Per-agent relative distance (log)", yscale="log")

    # 2. Per-agent relative error (linear)
    _plot_per_agent_metric(axes[1], iters, {ag: e_rel_df[ag].values for ag in agents}, "Relative distance",
                           "Per-agent relative distance (linear)")

    # 3. Per-agent absolute error (linear)
    _plot_per_agent_metric(axes[2], iters, {ag: e_abs_df[ag].values for ag in agents}, "Distance",
                           "Per-agent distance to mean (abs)")
    # 4. Average absolute error
    axes[3].plot(iters, avg_df["E_abs_mean"], label="Mean (abs)", linewidth=2)
    axes[3].set_xlabel("Round")
    axes[3].set_ylabel("Average distance")
    axes[3].set_title("Average distance to mean (abs)")
    axes[3].legend()

    # 5. Average relative error
    axes[4].plot(iters, avg_df["E_rel_mean"], label="Mean (rel)", linewidth=2)
    axes[4].set_xlabel("Round")
    axes[4].set_ylabel("Average relative distance")
    axes[4].set_title("Average relative distance to mean")
    axes[4].legend()

    # 6. Diameter
    axes[5].plot(iters, avg_df["D_rel"], label="Normalized", linewidth=2)
    axes[5].plot(iters, avg_df["D_abs"], label="Absolute", linestyle="--", linewidth=2)
    axes[5].set_xlabel("Round")
    axes[5].set_ylabel("Max pairwise distance")
    axes[5].set_title("Max pairwise distance between agents")
    axes[5].legend()

    # 7. Potential
    axes[6].plot(iters, avg_df["V_rel"], label="Normalized", linewidth=2)
    axes[6].plot(iters, avg_df["V_abs"], label="Absolute", linestyle="--", linewidth=2)
    axes[6].set_yscale("log")
    axes[6].set_xlabel("Round")
    axes[6].set_ylabel("Sum of squared distances")
    axes[6].set_title("Total squared distance to mean")
    axes[6].legend()

    # Hide unused subplots
    for idx in range(7, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Consensus metrics: {run_name}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig_path = out_dir / "consensus_all_metrics.png"
    plt.savefig(fig_path)
    plt.close(fig)

    # Save CSVs
    p_e_rel = out_dir / "E_rel_per_agent.csv"
    p_e_abs = out_dir / "E_abs_per_agent.csv"
    e_rel_df.to_csv(p_e_rel, index=False)
    e_abs_df.to_csv(p_e_abs, index=False)
    p_avg = out_dir / "consensus_summary.csv"
    avg_df.to_csv(p_avg, index=False)

    return {
        "E_rel_per_agent_csv": str(p_e_rel),
        "E_abs_per_agent_csv": str(p_e_abs),
        "summary_csv": str(p_avg),
        "all_metrics_plot": str(fig_path),
    }

def consensus(run_dir: Path) -> dict[str, str]:
    """
    Generate a consensus summary of all agents in the given run directory.

    Args:
        run_dir (Path): The directory containing agent subdirectories.
    """
    out_dir = run_dir / "consensus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if consensus has already been computed
    existing_files = ['E_rel_per_agent.csv', 'E_abs_per_agent.csv', 'consensus_summary.csv', 'consensus_all_metrics.png']
    if all((out_dir / fname).exists() for fname in existing_files):
        print(f"Consensus metrics already exist in {out_dir}, skipping computation.")
        return {
            "E_rel_per_agent_csv": str(out_dir / 'E_rel_per_agent.csv'),
            "E_abs_per_agent_csv": str(out_dir / 'E_abs_per_agent.csv'),
            "summary_csv": str(out_dir / 'consensus_summary.csv'),
            "all_metrics_plot": str(out_dir / 'consensus_all_metrics.png'),
            "per_layer_plot": str(out_dir / 'consensus_per_layer_all_layers.png'),
        }

    run_name = extract_run_info(run_dir)
    res = get_consensus_metrics(run_dir)
    per_layer_res = plot_per_layer_consensus(run_dir, out_dir)
    total_res = plot_total_consensus(res, out_dir, run_name)
    return {**per_layer_res, **total_res}
