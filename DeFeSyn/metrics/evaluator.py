import argparse
import glob
import re
import sys
from pathlib import Path
from typing import List, Dict, Union, Tuple

import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.spade.start import ADULT_MANIFEST, ADULT_PATH
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


class SynEvaluator:
    """
        Dataset evaluator for synthetic data.:
        - Loads train/test via DatasetLoader(manifest_path)
        - Loads pickled model and samples |train| synthetic rows
        - Drops NaNs and (if exists) filters workclass != 'Never-worked'
        - Runs selected metrics and saves results under output_dir
        Supported metric names:
            Privacy: DCR, NNDR, AdversarialAccuracy, SinglingOut, Inference, Linkability, Disclosure
            Utility: BasicStats, JS, KS, CorrelationPearson, CorrelationSpearman,
            Extras: PCA
            Consensus Metrics: Consensus
        """
    def __init__(self, manifest_path: str, model_path: str, output_dir: str,
                 metrics: List[str] = None, keys: List[str] = None, target: str = None,
                 inf_aux_cols: List[str] = None, secret: str = None, regression: bool = False,
                 link_aux_cols: Tuple[List[str], List[str]] = None, control_frac: float = 0.3,
                 original_name: str = "adult", synthetic_name: str = "CTGAN",
                 run_dir: str = None):
        self.seed = 42
        self.manifest_path = manifest_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_name = original_name
        self.synthetic_name = synthetic_name
        self.run_dir = Path(run_dir) if run_dir else None

        self.metrics = metrics if metrics is not None else [
            "DCR", "NNDR", "AdversarialAccuracy", "SinglingOut", "Inference", "Linkability", "Disclosure",
            "BasicStats", "JS", "KS",
            "CorrelationPearson", "CorrelationSpearman", "PCA",
            "Consensus"
        ]

        # Disclosure
        self.keys = keys
        self.target = target
        if "Disclosure" in self.metrics and (self.keys is None or self.target is None):
            raise ValueError("Keys and target must be provided for Disclosure metric.")

        # Inference
        self.inf_aux_cols = inf_aux_cols
        self.secret = secret
        self.regression = regression
        if "Inference" in self.metrics and (self.inf_aux_cols is None or self.secret is None):
            raise ValueError("Auxiliary columns and secret must be provided for Inference metric.")

        # Linkability
        self.link_aux_cols = link_aux_cols
        self.control_frac = control_frac
        if "Linkability" in self.metrics and self.link_aux_cols is None:
            raise ValueError("Auxiliary columns must be provided for Linkability metric.")

    def evaluate(self) -> Dict[str, Dict[str, Union[float, dict, str]]]:
        loader = DatasetLoader(self.manifest_path)
        full_train = loader.get_train()
        model = load_model_pickle(Path(self.model_path))
        synthetic = model.sample(len(full_train), self.seed)

        # simple clean
        full_train = full_train.dropna().reset_index(drop=True)
        synthetic = synthetic.dropna().reset_index(drop=True)
        if "workclass" in full_train.columns:
            full_train = full_train[full_train["workclass"] != "Never-worked"]
        if "workclass" in synthetic.columns:
            synthetic = synthetic[synthetic["workclass"] != "Never-worked"]

        privacy_res, utility_res, artifacts = {}, {}, {}

        # ===================== PRIVACY METRICS =====================
        print("Running privacy metrics...")
        for name in self.metrics:
            print(name)
            if name == "DCR":
                r = self._eval_privacy(DCRCalculator(original=full_train, synthetic=synthetic,
                                                     original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)
            if name == "NNDR":
                r = self._eval_privacy(NNDRCalculator(original=full_train, synthetic=synthetic,
                                                     original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "AdversarialAccuracy":
                r = self._eval_privacy(
                    AdversarialAccuracyCalculator(original=full_train, synthetic=synthetic,
                                                  original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "SinglingOut":
                r = self._eval_privacy(SinglingOutCalculator(original=full_train, synthetic=synthetic,
                                                             original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "Inference":
                r = self._eval_privacy(
                    InferenceCalculator(original=full_train, synthetic=synthetic, aux_cols=self.inf_aux_cols,
                                        secret=self.secret, regression=self.regression,
                                        original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "Linkability":
                control_frac = 0.3
                link_control_df = full_train.sample(frac=control_frac, random_state=self.seed)
                link_original_train_df = full_train.drop(link_control_df.index).reset_index(drop=True)
                link_control_df = link_control_df.reset_index(drop=True)
                r = self._eval_privacy(LinkabilityCalculator(original=link_original_train_df, synthetic=synthetic,
                                                            aux_cols=self.link_aux_cols, control=link_control_df,
                                                            original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "Disclosure":
                r = self._eval_privacy(
                    DisclosureCalculator(original=full_train, synthetic=synthetic, keys=self.keys, target=self.target,
                                        original_name=self.original_name, synthetic_name=self.synthetic_name))
                privacy_res[name] = r
                self._save_txt(name, r)
                print(r)


        # ===================== UTILITY METRICS =====================
        print("Running utility metrics...")
        for name in self.metrics:
            print(name)
            if name == "BasicStats":
                r = self._eval_utility(
                    BasicStatsCalculator(original=full_train, synthetic=synthetic,
                                         original_name=self.original_name, synthetic_name=self.synthetic_name))
                utility_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "JS":
                r = self._eval_utility(JSCalculator(original=full_train, synthetic=synthetic,
                                                    original_name=self.original_name, synthetic_name=self.synthetic_name))
                utility_res[name] = r
                self._save_txt(name, r)
                print(r)
            elif name == "KS":
                r = self._eval_utility(KSCalculator(original=full_train, synthetic=synthetic,
                                                    original_name=self.original_name, synthetic_name=self.synthetic_name))
                utility_res[name] = r
                self._save_txt(name, r)
                print(r)

        # ===================== EXTRAS =====================
        if "CorrelationPearson" in self.metrics:
            print("  CorrelationPearson...")
            artifacts.update(
                self._correlation(full_train, synthetic, CorrelationMethod.PEARSON, "CorrelationPearson"))
        if "CorrelationSpearman" in self.metrics:
            print("  CorrelationSpearman...")
            artifacts.update(
                self._correlation(full_train, synthetic, CorrelationMethod.SPEARMAN, "CorrelationSpearman"))
        if "PCA" in self.metrics:
            print("  PCA...")
            artifacts["PCA"] = str(self._pca(full_train, synthetic))
        if "Consensus" in self.metrics:
            print("  Consensus...")
            if not self.run_dir:
                raise ValueError(
                    "Consensus metric requires --run-dir pointing to the run folder that contains agent_* subdirs.")
            artifacts["Consensus"] = self._consensus(self.run_dir)

        result =  {"privacy": privacy_res, "utility": utility_res, "artifacts": artifacts}
        p = self.output_dir / "results.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(result))
        return result

    # --- helpers ---
    @staticmethod
    def _eval_privacy(metric) -> Union[float, dict]:
        m = PrivacyMetricManager()
        m.add_metric(metric)
        return m.evaluate_all()

    @staticmethod
    def _eval_utility(metric) -> Union[float, dict]:
        m = UtilityMetricManager()
        m.add_metric(metric)
        return m.evaluate_all()

    # ===================== CORRELATION METRIC =====================

    def _correlation(self, full_train, synthetic, method: CorrelationMethod, label: str) -> Dict[str, str]:
        metric = CorrelationCalculator(full_train, synthetic, self.original_name, self.synthetic_name)
        score = metric.evaluate(method=method)
        with open(self.output_dir / f"{label}_result.txt", "w") as f:
            f.write(f"{label}: {score}\n")

        orig_df, syn_df = metric.correlation_pairs(method=method)
        p_orig_csv = self.output_dir / f"{label}_original_correlation.csv"
        p_syn_csv = self.output_dir / f"{label}_synthetic_correlation.csv"
        orig_df.to_csv(p_orig_csv, index=False)
        syn_df.to_csv(p_syn_csv, index=False)

        # quick heatmaps
        plt.figure(figsize=(9, 7))
        sns.heatmap(orig_df, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Original ({method.name.title()})")
        p1 = self.output_dir / f"{label}_original_heatmap.png"
        plt.savefig(p1)
        plt.clf()
        plt.figure(figsize=(9, 7))
        sns.heatmap(syn_df, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Synthetic ({method.name.title()})")
        p2 = self.output_dir / f"{label}_synthetic_heatmap.png"
        plt.savefig(p2)
        plt.clf()

        return {
            f"{label}_score": str(self.output_dir / f"{label}_result.txt"),
            f"{label}_orig_csv": str(p_orig_csv),
            f"{label}_syn_csv": str(p_syn_csv),
            f"{label}_orig_heatmap": str(p1),
            f"{label}_syn_heatmap": str(p2),
        }

    # ===================== PCA METRIC =====================

    def _pca(self, full_train, synthetic) -> Path:
        from sklearn.decomposition import PCA
        combined = pd.concat([full_train, synthetic], ignore_index=True)
        X = combined.select_dtypes(include=[np.number]).fillna(0)
        pca = PCA(n_components=2).fit_transform(X)
        n = len(full_train)
        plt.figure(figsize=(9, 7))
        plt.scatter(pca[:n, 0], pca[:n, 1], label="Original", alpha=0.5)
        plt.scatter(pca[n:, 0], pca[n:, 1], label="Synthetic", alpha=0.5)
        plt.title("PCA: Original vs Synthetic")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        out = self.output_dir / "PCA_Original_vs_Synthetic.png"
        plt.savefig(out)
        plt.clf()
        return out

    # ===================== CONSENSUS METRIC (model-parameter-level) =====================

    _ITER_RE = re.compile(r"iter-(\d+)-weights\.pt$")

    @staticmethod
    def _load_vec(path: Path) -> np.ndarray:
        sd = torch.load(path, map_location="cpu")

        def walk(v):
            if isinstance(v, dict):
                for x in v.values():
                    yield from walk(x)
            else:
                try:
                    arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                    if hasattr(arr, "dtype") and np.issubdtype(arr.dtype, np.number):
                        yield arr.ravel()
                except Exception:
                    pass

        parts = [p for p in walk(sd) if p.size > 0]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

    def _consensus_collect(self, run_dir: Path):
        pattern = run_dir / "agent_*" / "iter-*-weights.pt"
        files = sorted(glob.glob(str(pattern)))
        by_agent = {}
        for p in files:
            pth = Path(p)
            ag = pth.parent.name
            m = self._ITER_RE.search(pth.name)
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
        vecs = {ag: [self._load_vec(by_agent[ag][t]) for t in common_iters] for ag in agents}

        # Check equal parameter vector shapes across agents (optional strictness)
        ref_len = {len(v[0]) for v in vecs.values()}
        if len(ref_len) != 1:
            # Not fatal, but this usually indicates inconsistent state dicts
            print("WARNING: Not all agents have identical parameter vector lengths.", file=sys.stderr)

        return agents, common_iters, vecs

    def _consensus_curves_full(self, run_dir: Path, eps: float = 1e-12):
        """
        Compute consensus metrics (per-agent error E, diameter D, potential V)
        in both absolute and normalized forms.

        Returns a dict with:
          agents, iters, E_abs/E_rel (dict of lists), D_abs/D_rel (lists), V_abs/V_rel (lists)
        """
        agents, iters, vecs = self._consensus_collect(run_dir)

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

            # Diameter
            dmax_abs = 0.0
            for i in range(n_agents):
                Xi = X[i]
                for j in range(i + 1, n_agents):
                    d_ij = float(np.linalg.norm(Xi - X[j]))
                    if d_ij > dmax_abs:
                        dmax_abs = d_ij
            D_abs.append(dmax_abs)
            D_rel.append(dmax_abs / mu_norm_safe)

            # Potential (sum of squares)
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

    def _consensus(self, run_dir: Path) -> Dict[str, str]:
        """
        Run consensus metric, save artifacts (plots + CSVs) under output_dir/Consensus,
        and return a dict of artifact paths.
        """
        out_dir = self.output_dir / "Consensus"
        out_dir.mkdir(parents=True, exist_ok=True)

        res = self._consensus_curves_full(run_dir)
        agents, iters = res["agents"], np.asarray(res["iters"])

        # ---------- Save CSV summaries ----------
        # Per-agent E_rel & E_abs
        e_rel_df = pd.DataFrame({"iter": iters})
        e_abs_df = pd.DataFrame({"iter": iters})
        for ag in agents:
            e_rel_df[ag] = res["E_rel"][ag]
            e_abs_df[ag] = res["E_abs"][ag]
        p_e_rel = out_dir / "E_rel_per_agent.csv"
        p_e_abs = out_dir / "E_abs_per_agent.csv"
        e_rel_df.to_csv(p_e_rel, index=False)
        e_abs_df.to_csv(p_e_abs, index=False)

        # Averages (mean across agents)
        Erel_mat = e_rel_df[agents].to_numpy()
        Eabs_mat = e_abs_df[agents].to_numpy()
        avg_df = pd.DataFrame({
            "iter": iters,
            "E_rel_mean": Erel_mat.mean(axis=1),
            "E_rel_rms":  np.sqrt((Erel_mat**2).mean(axis=1)),
            "E_abs_mean": Eabs_mat.mean(axis=1),
            "E_abs_rms":  np.sqrt((Eabs_mat**2).mean(axis=1)),
            "D_rel": res["D_rel"],
            "D_abs": res["D_abs"],
            "V_rel": res["V_rel"],
            "V_abs": res["V_abs"],
        })
        p_avg = out_dir / "consensus_summary.csv"
        avg_df.to_csv(p_avg, index=False)

        # ---------- PLOTS ----------
        # 1) Per-agent relative error
        fig1 = out_dir / "consensus-error-per-agent.png"
        plt.figure()
        for ag in agents:
            plt.plot(iters, res["E_rel"][ag], label=ag, alpha=0.9)
        plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Relative consensus error")
        plt.title(f"Per-agent consensus error (relative). Run: {run_dir.name}")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(fig1)
        plt.clf()

        # 2) Average errors
        fig2 = out_dir / "consensus-error-average.png"
        plt.figure()
        plt.plot(iters, avg_df["E_rel_mean"], label="Mean (rel)", linewidth=2)
        plt.plot(iters, avg_df["E_abs_mean"], label="Mean (abs)", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Consensus error")
        plt.title(f"Average consensus error. {run_dir.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig2)
        plt.clf()

        # 3) Diameter
        fig3 = out_dir / "consensus-diameter.png"
        plt.figure()
        plt.plot(iters, avg_df["D_rel"], label="Normalized", linewidth=2)
        plt.plot(iters, avg_df["D_abs"], label="Absolute", linestyle="--", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Diameter (max pairwise dist)")
        plt.title(f"Network diameter. {run_dir.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig3)
        plt.clf()

        # 4) Potential
        fig4 = out_dir / "consensus-potential.png"
        plt.figure()
        plt.plot(iters, avg_df["V_rel"], label="Normalized", linewidth=2)
        plt.plot(iters, avg_df["V_abs"], label="Absolute", linestyle="--", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Potential Σ||θ_i−μ||²")
        plt.title(f"Consensus potential. {run_dir.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig4)
        plt.clf()

        # Return artifact paths (for inclusion in overall results)
        return {
            "E_rel_per_agent_csv": str(p_e_rel),
            "E_abs_per_agent_csv": str(p_e_abs),
            "summary_csv": str(p_avg),
            "per_agent_plot": str(fig1),
            "avg_error_plot": str(fig2),
            "diameter_plot": str(fig3),
            "potential_plot": str(fig4),
        }


    def _save_txt(self, metric_name: str, result: Union[float, dict]) -> None:
        p = self.output_dir / f"{metric_name}_result.txt"
        with open(p, "w", encoding="utf-8") as f:
            if isinstance(result, dict):
                for k, v in result.items(): f.write(f"{k}: {v}\n")
            else:
                f.write(f"value: {result}\n")

def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []

def cli(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data privacy & utility metrics."
    )

    # Required-ish (with sensible defaults)
    parser.add_argument("--manifest-path", required=True,
                        help="Path to dataset manifest (e.g., ADULT_MANIFEST).")
    parser.add_argument("--model-path", required=True,
                        help="Path to pickled generative model (.pkl).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write results/artifacts.")

    # Labels
    parser.add_argument("--original-name", default="adult",
                        help="Name label for the original dataset.")
    parser.add_argument("--synthetic-name", default="CTGAN",
                        help="Name label for the synthetic dataset.")

    # Metrics
    parser.add_argument(
        "--metrics",
        type=_csv_list,
        default=["DCR", "NNDR", "AdversarialAccuracy", "SinglingOut", "Inference",
                 "Linkability", "Disclosure", "BasicStats", "JS", "KS",
                 "CorrelationPearson", "CorrelationSpearman", "PCA", "Consensus"],
        help=("Comma-separated metric names to run. Supported: "
              "DCR, NNDR, AdversarialAccuracy, SinglingOut, Inference, Linkability, Disclosure, "
              "BasicStats, JS, KS, CorrelationPearson, CorrelationSpearman, PCA")
    )

    # Disclosure
    parser.add_argument("--keys", type=_csv_list, default=[],
                        help="Comma-separated quasi-identifier keys for Disclosure.")
    parser.add_argument("--target", default=None,
                        help="Target/secret column for Disclosure.")

    # Inference
    parser.add_argument("--inf-aux-cols", type=_csv_list, default=[],
                        help="Comma-separated auxiliary columns for Inference.")
    parser.add_argument("--secret", default=None,
                        help="Secret column (target) for Inference.")
    parser.add_argument("--regression", action="store_true",
                        help="Treat Inference secret as regression.")

    # Linkability
    parser.add_argument("--link-a", type=_csv_list, default=[],
                        help="Comma-separated set A auxiliary columns for Linkability.")
    parser.add_argument("--link-b", type=_csv_list, default=[],
                        help="Comma-separated set B auxiliary columns for Linkability.")
    parser.add_argument("--control-frac", type=float, default=0.3,
                        help="Control fraction for Linkability (0-1).")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Consensus
    parser.add_argument("--run-dir", default=None,
                        help="Path to run directory containing agent_* subfolders with iter-*-weights.pt (required for Consensus).")

    args = parser.parse_args(argv)

    # Validate conditional requirements
    orig_metrics = args.metrics
    selected = set([m.lower() for m in args.metrics])

    def want(name: str) -> bool:
        return name.lower() in selected

    # Build linkability aux tuple if needed
    link_aux_cols = None
    if want("linkability"):
        if not args.link_a or not args.link_b:
            print("ERROR: --link-a and --link-b are required when running Linkability.", file=sys.stderr)
            return 2
        link_aux_cols = (args.link_a, args.link_b)

    # Disclosure requirements
    if want("disclosure"):
        if not args.keys or not args.target:
            print("ERROR: --keys and --target are required when running Disclosure.", file=sys.stderr)
            return 2

    # Inference requirements
    if want("inference"):
        if not args.inf_aux_cols or not args.secret:
            print("ERROR: --inf-aux-cols and --secret are required when running Inference.", file=sys.stderr)
            return 2

    # Consensus requirements
    if want("consensus"):
        if not args.run_dir:
            print("ERROR: --run-dir is required when running Consensus.", file=sys.stderr)
            return 2

    evaluator = SynEvaluator(
        manifest_path=args.manifest_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        original_name=args.original_name,
        synthetic_name=args.synthetic_name,
        metrics=orig_metrics if args.metrics else None,
        keys=args.keys if args.keys else None,
        target=args.target,
        inf_aux_cols=args.inf_aux_cols if args.inf_aux_cols else None,
        secret=args.secret,
        regression=bool(args.regression),
        link_aux_cols=link_aux_cols,
        control_frac=args.control_frac,
        run_dir=args.run_dir
    )

    # Optional: set seed if you also want to seed numpy/py modules consistently inside evaluate()
    evaluator.seed = args.seed

    results = evaluator.evaluate()
    print("\n========= SUMMARY =========")
    print(results)
    return 0

if __name__ == "__main__":
    raise SystemExit(cli())
