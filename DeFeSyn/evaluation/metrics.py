import glob, re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from DeFeSyn.io.io import get_repo_root

ITER_RE = re.compile(r"iter-(\d+)-weights\.pt$")

def load_vec(path):
    sd = torch.load(path, map_location="cpu")
    # flatten: concatenate all numeric leaves (dicts may be nested)
    def walk(v):
        if isinstance(v, dict):
            for x in v.values(): yield from walk(x)
        else:
            try:
                arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                if hasattr(arr, "dtype") and np.issubdtype(arr.dtype, np.number):
                    yield arr.ravel()
            except Exception:
                pass
    parts = [p for p in walk(sd) if p.size > 0]
    return np.concatenate(parts, axis=0)

def collect(run_dir):
    pattern = Path(run_dir) / "agent_*" / "iter-*-weights.pt"
    files = sorted(glob.glob(str(pattern)))
    by_agent = {}
    for p in files:
        ag = Path(p).parent.name
        m = ITER_RE.search(p)
        if not m: continue
        t = int(m.group(1))
        by_agent.setdefault(ag, {})[t] = p
    # align on common iterations
    iters = sorted(set.intersection(*(set(d.keys()) for d in by_agent.values())))
    agents = sorted(by_agent.keys())
    vecs = {ag: [load_vec(by_agent[ag][t]) for t in iters] for ag in agents}
    return agents, iters, vecs

def consensus_curves(run_dir, eps=1e-12):
    agents, iters, vecs = collect(run_dir)
    # stack per t for mean/diameter/variance
    E = {ag: [] for ag in agents}
    D, V = [], []
    for k, t in enumerate(iters):
        X = np.stack([vecs[ag][k] for ag in agents], axis=0)
        mu = X.mean(axis=0)
        mu_norm = np.linalg.norm(mu) + eps
        # per-agent error
        for i, ag in enumerate(agents):
            E[ag].append(np.linalg.norm(X[i] - mu)/mu_norm)
        # diameter & potential
        dmax = 0.0
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                dmax = max(dmax, np.linalg.norm(X[i]-X[j]) / mu_norm)
        D.append(dmax)
        V.append(((X - mu) ** 2).sum())
    return agents, iters, E, D, V

def consensus_curves_full(run_dir, eps: float = 1e-12):
    """
    Compute consensus metrics (per-agent error E, diameter D, potential V)
    in both unnormalized (absolute) and normalized forms.

    Normalizations:
      - E_rel[i,k] = ||w_i - mu|| / max(||mu||, eps)
      - D_rel[k]   = max_{i<j} ||w_i - w_j|| / max(||mu||, eps)
      - V_rel[k]   = sum_i ||w_i - mu||^2 / (n * max(||mu||^2, eps^2))

    Returns a dict:
      {
        'agents': List[str],
        'iters':  List[int],
        'E_abs':  Dict[agent, List[float]],  # per-agent absolute errors
        'E_rel':  Dict[agent, List[float]],  # per-agent normalized errors
        'D_abs':  List[float],               # absolute diameter
        'D_rel':  List[float],               # normalized diameter
        'V_abs':  List[float],               # absolute potential (sum of squares)
        'V_rel':  List[float],               # normalized potential
      }
    """
    agents, iters, vecs = collect(run_dir)

    E_abs = {ag: [] for ag in agents}
    E_rel = {ag: [] for ag in agents}
    D_abs, D_rel = [], []
    V_abs, V_rel = [], []

    n_agents = len(agents)

    for k, t in enumerate(iters):
        # Stack all agents' parameter vectors at iteration k: shape (n_agents, n_params)
        X = np.stack([vecs[ag][k] for ag in agents], axis=0)

        # Mean model and its norm
        mu = X.mean(axis=0)
        mu_norm = np.linalg.norm(mu)
        mu_norm_safe = max(mu_norm, eps)

        # Deviations from mean (shape like X)
        diffs = X - mu

        # --- E: per-agent error (abs & rel) ---
        for i, ag in enumerate(agents):
            e_abs = np.linalg.norm(diffs[i])
            E_abs[ag].append(e_abs)
            E_rel[ag].append(e_abs / mu_norm_safe)

        # --- D: diameter (max pairwise distance) ---
        dmax_abs = 0.0
        for i in range(n_agents):
            Xi = X[i]
            for j in range(i + 1, n_agents):
                d_ij = np.linalg.norm(Xi - X[j])
                if d_ij > dmax_abs:
                    dmax_abs = d_ij
        D_abs.append(dmax_abs)
        D_rel.append(dmax_abs / mu_norm_safe)

        # --- V: potential (sum of squared deviations from mean) ---
        v_abs = float(np.sum(diffs ** 2))
        V_abs.append(v_abs)

        # Normalize V by n * ||mu||^2 (scale- and size-invariant)
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


# --- plotting
def plot_consensus(run_dir):
    agents, iters, E, D, V = consensus_curves(run_dir)
    res = consensus_curves_full()
    # per-agent error
    fig_name = Path(run_dir) / "consensus-error.png"
    plt.figure()
    for ag in agents:
        plt.plot(iters, E[ag], label=ag, alpha=0.9)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Consensus error ||θ_i-μ||/||μ||")
    plt.title("Per-agent consensus error. Run: " + Path(run_dir).name)
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_name)

    # diameter
    fig_name = Path(run_dir) / "consensus-diameter.png"
    plt.figure()
    plt.plot(iters, D)
    plt.yscale("log")
    plt.xlabel("Round"); plt.ylabel("Normalized diameter (max pairwise dist)")
    plt.title("Network diameter over rounds. Run: " + Path(run_dir).name)
    plt.tight_layout()
    plt.savefig(fig_name)

    # potential (variance)
    fig_name = Path(run_dir) / "consensus-potential.png"
    plt.figure()
    plt.plot(iters, V)
    plt.yscale("log")
    plt.xlabel("Round"); plt.ylabel("Consensus potential Σ||θ_i-μ||²")
    plt.title("Consensus potential. Run: " + Path(run_dir).name)
    plt.tight_layout()
    plt.savefig(fig_name)


def plot_consensus_full(run_dir):
    res = consensus_curves_full(run_dir)  # returns dict with E_abs/E_rel/etc.
    agents, iters = res["agents"], np.asarray(res["iters"])
    n = len(agents)

    def stack(metric_dict):
        """Stack per-agent dict -> array of shape (n_agents, T)."""
        return np.stack([metric_dict[ag] for ag in agents], axis=0)

    Erel = stack(res["E_rel"])
    Eabs = stack(res["E_abs"])

    # ------------------------------------------------
    # 1) Per-agent plots (same as before)
    # ------------------------------------------------
    fig_name = Path(run_dir) / "consensus-error-per-agent.png"
    plt.figure()
    for ag in agents:
        plt.plot(iters, res["E_rel"][ag], label=ag, alpha=0.9)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Relative consensus error")
    plt.title(f"Per-agent consensus error (relative). Run: {Path(run_dir).name}")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(fig_name)

    # ------------------------------------------------
    # 2) Average-only plots
    # ------------------------------------------------
    # You can choose mean or RMS; here I show both
    mean_rel = Erel.mean(axis=0)
    rms_rel  = np.sqrt((Erel**2).mean(axis=0))

    mean_abs = Eabs.mean(axis=0)
    rms_abs  = np.sqrt((Eabs**2).mean(axis=0))

    fig_name = Path(run_dir) / "consensus-error-average.png"
    plt.figure()
    plt.plot(iters, mean_rel, label="Mean (rel)", linewidth=2)
    #plt.plot(iters, rms_rel,  label="RMS (rel)",  linewidth=2, linestyle="--")
    plt.plot(iters, mean_abs, label="Mean (abs)", linewidth=2, color="C2")
    #plt.plot(iters, rms_abs,  label="RMS (abs)",  linewidth=2, linestyle="--", color="C3")
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Consensus error")
    plt.title(f"Average consensus error. {Path(run_dir).name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name)

    # ------------------------------------------------
    # Diameter & Potential plots as before
    # ------------------------------------------------
    fig_name = Path(run_dir) / "consensus-diameter.png"
    plt.figure()
    plt.plot(iters, res["D_rel"], label="Normalized", linewidth=2)
    plt.plot(iters, res["D_abs"], label="Absolute", linestyle="--", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Diameter (max pairwise dist)")
    plt.title(f"Network diameter. {Path(run_dir).name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name)

    fig_name = Path(run_dir) / "consensus-potential.png"
    plt.figure()
    plt.plot(iters, res["V_rel"], label="Normalized", linewidth=2)
    plt.plot(iters, res["V_abs"], label="Absolute", linestyle="--", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Potential Σ||θ_i−μ||²")
    plt.title(f"Consensus potential. {Path(run_dir).name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name)



if __name__ == "__main__":
    # Example usage:
    root = get_repo_root()
    run_id = "run-20250831-212602-a4-e2-i150-alpha1.0-ring"
    run_dir = root / "runs"
    rd = run_dir / run_id
    print(f"Processing {rd.name}...")
    plot_consensus_full(rd)

    # Iterate over all runs in the runs directory
    # run_dirs = sorted(run_dir.glob("run-*"))
    # for rd in run_dirs:
    #     print(f"Processing {rd.name}...")
    #     plot_consensus_full(rd)
