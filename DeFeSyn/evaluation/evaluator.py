import argparse
import glob
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import defaultdict

from DeFeSyn.data.data_loader import DatasetLoader
from DeFeSyn.io.io import load_model_pickle, get_config_dir
from DeFeSyn.models.tab_ddpm.lib import load_config
from DeFeSyn.models.tab_ddpm.scripts.sample import sample
from FEST.privacy_utility_framework.build.lib.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import \
    KSCalculator
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
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.privacy_metric_manager import \
    PrivacyMetricManager
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.basic_stats import \
    BasicStatsCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.correlation import \
    CorrelationMethod, CorrelationCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.js_similarity import \
    JSCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.wasserstein import \
    WassersteinCalculator, WassersteinMethod
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.utility_metric_manager import \
    UtilityMetricManager

def _is_tensor(x):
    return isinstance(x, torch.Tensor)

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

class Evaluator:
    def __init__(
            self,
            data_dir: str,
            categorical_cols: list[str],
            model_path: str,
            output_dir: str,
            metrics: list[str] = None,
            keys: list[str] = None,
            target: str = None,
            inf_aux_cols: list[str] = None,
            secret: str = None,
            regression: bool = False,
            link_aux_cols: tuple[list[str], list[str]] = None,
            control_frac: float = 0.3,
            original_name: str = "adult",
            synthetic_name: str = "CTGAN",
            model_type: str = "ctgan",
            run_dir: str = None,
    ):
        self.seed = 42

        self.data_dir = data_dir
        self.categorical_cols = categorical_cols

        self.model_type = model_type
        self.model_path = model_path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.privacy_dir = self.output_dir / "Privacy"
        self.privacy_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_dir = self.output_dir / "Similarity"
        self.similarity_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir) if run_dir else None

        self.original_name = original_name
        self.synthetic_name = synthetic_name

        self.results = pd.DataFrame(columns=[
            "DCR", "NNDR", "AdversarialAccuracy", "SinglingOut", "Inference", "Linkability", "Disclosure",
            "RepU", "DiSCO",
            "Mean", "Median", "Variance", "JS", "KS", "WASSERSTEIN",
            "CorrelationPearson", "CorrelationSpearman", "PCA",
            "Consensus"
        ], index=[self.synthetic_name])

        self.metrics = metrics if metrics is not None else [
            "DCR", "NNDR", "AdversarialAccuracy", "Disclosure",
            "BasicStats", "JS", "KS",
            "CorrelationPearson", "CorrelationSpearman", "PCA", "PCA3D"
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

        self.metrics = self.filter_metrics()

    def filter_metrics(self) -> list[str]:
        filtered_metrics = []
        for metric in self.metrics:
            metric_lower = metric.lower()
            if metric_lower in ["dcr", "nndr", "adversarialaccuracy", "singlingout", "inference", "linkability",
                                "disclosure"]:
                result_file = self.privacy_dir / f"{metric}_result.txt"
                # Disclosure writes two files
                if metric_lower == "disclosure":
                    repu_file = self.privacy_dir / "RepU_result.txt"
                    disco_file = self.privacy_dir / "DiSCO_result.txt"
                    if repu_file.exists() and disco_file.exists():
                        continue
                elif result_file.exists():
                    continue
            elif metric_lower in ["basicstats", "js", "ks", "wasserstein"]:
                result_file = self.similarity_dir / f"{metric}_result.txt"
                if result_file.exists():
                    continue
            elif metric_lower == "correlationpearson":
                result_file = self.similarity_dir / "CorrelationPearson_result.txt"
                if result_file.exists():
                    continue
            elif metric_lower == "correlationspearman":
                result_file = self.similarity_dir / "CorrelationSpearman_result.txt"
                if result_file.exists():
                    continue
            elif metric_lower == "pca":
                result_file = self.similarity_dir / "PCA_Original_vs_Synthetic.png"
                if result_file.exists():
                    continue
            elif metric_lower == "pca3d":
                result_file = self.similarity_dir / "PCA3D_Original_vs_Synthetic.html"
                if result_file.exists():
                    continue
            elif metric_lower == "consensus":
                result_file = self.similarity_dir / "Consensus_result.txt"
                if result_file.exists():
                    continue
            filtered_metrics.append(metric)
        return filtered_metrics

    def evaluate(self) -> dict:
        if self.metrics == []:
            print("All specified metrics have existing results. Skipping evaluation.")
            return {}

        loader = DatasetLoader(self.data_dir, self.categorical_cols)
        original_data = loader.get_train()



        if self.model_type.lower() == "tabddpm" and self.metrics != ['Consensus']:
            # Convert all integer columns in full_train to float
            int_cols = original_data.select_dtypes(include=['int64']).columns
            original_data[int_cols] = original_data[int_cols].astype('float64')
            config = load_config(get_config_dir() / "config.toml")

            sample(
                parent_dir=self.run_dir,
                model_path=self.model_path,
                real_data_path=self.data_dir + "/npy",
                num_samples=10000,#len(original_data),
                batch_size=10000,#config['sample']['batch_size'],
                disbalance=config['sample'].get('disbalance', None),
                **config['diffusion_params'],
                model_type=config['model_type'],
                model_params=config['model_params'],
                T_dict=config['train']['T'],
                num_numerical_features=config['num_numerical_features'],
                device="cpu",
                seed=0,
                change_val=False
            )

            x_cat_p = self.run_dir / 'X_cat_train.npy'
            x_num_p = self.run_dir / 'X_num_train.npy'
            y_p = self.run_dir / 'y_train.npy'

            x_cat = np.load(x_cat_p, allow_pickle=True)
            x_num = np.load(x_num_p, allow_pickle=True)
            y = np.load(y_p, allow_pickle=True)


            x_gen = np.concatenate([x_num, x_cat], axis=1)
            synthetic = pd.DataFrame(x_gen, columns=original_data.columns.drop(self.target))
            synthetic[self.target] = np.asarray(y).squeeze()

            # Ensure categorical columns have the same dtype as original
            for col in self.categorical_cols:
                synthetic[col] = synthetic[col].astype(original_data[col].dtype)

            # Ensure numerical are float
            for col in original_data.columns:
                if col not in self.categorical_cols and col != self.target:
                    synthetic[col] = synthetic[col].astype('float64')
        elif self.model_type.lower() == "ctgan" and self.metrics != ['Consensus']:
            model = load_model_pickle(Path(self.model_path))
            #synthetic = model.sample(len(original_data), self.seed)
            synthetic = model.sample(10_000, self.seed)

        if self.metrics != ['Consensus']:
            privacy_res = self.privacy_metrics(original_data, synthetic)
            utility_res = self.utility_metrics(original_data, synthetic)
            artifacts = self.extras(original_data, synthetic)
        else:
            privacy_res = {}
            utility_res = {}
            artifacts = self.extras(original_data, None)

        results_path = os.path.join(self.output_dir, "results.csv")
        if os.path.exists(results_path):
            existing = pd.read_csv(results_path, index_col=0)
            self.results.update(existing)
        self.results.to_csv(results_path)
        result = {"privacy": privacy_res, "utility": utility_res, "artifacts": artifacts}
        p = self.output_dir / "results.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(result))
        return result


    def privacy_metrics(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
        privacy_res = {}
        print("Running privacy metrics...")
        for name in self.metrics:
            print(name)
            if name == "DCR":
                if (self.privacy_dir / "DCR_result.txt").exists():
                    print("DCR result already exists, skipping computation.")
                    continue
                r = DCRCalculator(original=original, synthetic=synthetic,
                                  original_name=self.original_name, synthetic_name=self.synthetic_name).evaluate()
                privacy_res[name] = r
                self._save_txt(name, r, self.privacy_dir)
                print(r)
                # Save to self.results dataframe. Row is model name, col is metric name
                self.results.at[self.synthetic_name, "DCR"] = r if isinstance(r, float) else r.get("DCR", np.nan)
            if name == "NNDR":
                if (self.privacy_dir / "NNDR_result.txt").exists():
                    print("NNDR result already exists, skipping computation.")
                    continue
                try:
                    r = NNDRCalculator(original=original, synthetic=synthetic,
                                       original_name=self.original_name, synthetic_name=self.synthetic_name).evaluate()
                except Exception as e:
                    print("NNDR calculation failed:", e)
                    continue
                privacy_res[name] = r
                self._save_txt(name, r, self.privacy_dir)
                self.results.at[self.synthetic_name, "NNDR"] = r if isinstance(r, float) else r.get("NNDR", np.nan)
                print(r)
            elif name == "AdversarialAccuracy":
                if (self.privacy_dir / "AdversarialAccuracy_result.txt").exists():
                    print("AdversarialAccuracy result already exists, skipping computation.")
                    continue
                try:
                    r = AdversarialAccuracyCalculator(original=original, synthetic=synthetic,
                                                      original_name=self.original_name,
                                                      synthetic_name=self.synthetic_name).evaluate()
                except Exception as e:
                    print("AdversarialAccuracy calculation failed:", e)
                    continue
                privacy_res[name] = r
                self._save_txt(name, r, self.privacy_dir)
                self.results.at[self.synthetic_name, "AdversarialAccuracy"] = r if isinstance(r, float) else r.get(
                    "AdversarialAccuracy", np.nan)
                print(r)
            elif name == "SinglingOut":
                if (self.privacy_dir / "SinglingOut_result.txt").exists():
                    print("SinglingOut result already exists, skipping computation.")
                    continue
                r = SinglingOutCalculator(original=original, synthetic=synthetic,
                                          original_name=self.original_name,
                                          synthetic_name=self.synthetic_name).evaluate()
                privacy_res[name] = r
                self.results.at[self.synthetic_name, "SinglingOut"] = r.value if isinstance(r.value, float) else r.get(
                    "SinglingOut", np.nan)
                self._save_txt(name, r, self.privacy_dir)
                print(r)
            elif name == "Inference":
                if (self.privacy_dir / "Inference_result.txt").exists():
                    print("Inference result already exists, skipping computation.")
                    continue
                r = InferenceCalculator(original=original, synthetic=synthetic, aux_cols=self.inf_aux_cols,
                                        secret=self.secret, regression=self.regression,
                                        original_name=self.original_name, synthetic_name=self.synthetic_name).evaluate()
                privacy_res[name] = r
                self.results.at[self.synthetic_name, "Inference"] = r.value if isinstance(r.value, float) else r.get(
                    "Inference", np.nan)
                self._save_txt(name, r, self.privacy_dir)
                print(r)
            elif name == "Linkability":
                if (self.privacy_dir / "Linkability_result.txt").exists():
                    print("Linkability result already exists, skipping computation.")
                    continue
                control_frac = 0.3
                link_control_df = original.sample(frac=control_frac, random_state=self.seed)
                link_original_train_df = original.drop(link_control_df.index).reset_index(drop=True)
                link_control_df = link_control_df.reset_index(drop=True)
                r = LinkabilityCalculator(original=link_original_train_df, synthetic=synthetic,
                                          aux_cols=self.link_aux_cols, control=link_control_df,
                                          original_name=self.original_name,
                                          synthetic_name=self.synthetic_name).evaluate()
                privacy_res[name] = r
                self.results.at[self.synthetic_name, "Linkability"] = r.value if isinstance(r.value, float) else r.get(
                    "Linkability", np.nan)
                self._save_txt(name, r, self.privacy_dir)
                print(r)
            elif name == "Disclosure":
                if (self.privacy_dir / "DiSCO_result.txt").exists() and (self.privacy_dir / "RepU_result.txt").exists():
                    print("Disclosure result already exists, skipping computation.")
                    continue
                discoCalc = DisclosureCalculator(original=original, synthetic=synthetic, keys=self.keys,
                                                 target=self.target,
                                                 original_name=self.original_name, synthetic_name=self.synthetic_name)
                res = discoCalc.evaluate()
                privacy_res[name] = res
                repU, disco = res
                self.results.at[self.synthetic_name, "RepU"] = repU
                self.results.at[self.synthetic_name, "DiSCO"] = disco
                self._save_txt("RepU", repU, self.privacy_dir)
                self._save_txt("DiSCO", disco, self.privacy_dir)
                print(res)

        return privacy_res

    def utility_metrics(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
        utility_res = {}
        print("Running utility metrics...")
        for name in self.metrics:
            if name == "BasicStats":
                if (self.similarity_dir / "BasicStats_result.txt").exists():
                    print("BasicStats result already exists, skipping computation.")
                    continue
                r = BasicStatsCalculator(original=original, synthetic=synthetic,
                                         original_name=self.original_name,
                                         synthetic_name=self.synthetic_name).evaluate()
                utility_res[name] = r
                mean = r['mean']
                median = r['median']
                variance = r['var']
                self.results.at[self.synthetic_name, "Mean"] = mean
                self.results.at[self.synthetic_name, "Variance"] = variance
                self.results.at[self.synthetic_name, "Median"] = median
                self._save_txt(name, r, self.similarity_dir)
                print(r)
            elif name == "JS":
                if (self.similarity_dir / "JS_result.txt").exists():
                    print("JS result already exists, skipping computation.")
                    continue
                r = JSCalculator(original=original, synthetic=synthetic,
                                 original_name=self.original_name, synthetic_name=self.synthetic_name).evaluate()
                utility_res[name] = r
                self.results.at[self.synthetic_name, "JS"] = r
                self._save_txt(name, r, self.similarity_dir)
                print(r)
            elif name == "KS":
                if (self.similarity_dir / "KS_result.txt").exists():
                    print("KS result already exists, skipping computation.")
                    continue
                r = KSCalculator(original=original, synthetic=synthetic,
                                 original_name=self.original_name, synthetic_name=self.synthetic_name).evaluate()
                utility_res[name] = r
                self.results.at[self.synthetic_name, "KS"] = r
                self._save_txt(name, r, self.similarity_dir)
                print(r)
            elif name == "WASSERSTEIN":
                if (self.similarity_dir / "WASSERSTEIN_result.txt").exists():
                    print("WASSERSTEIN result already exists, skipping computation.")
                    continue
                wasserstein = WassersteinCalculator(original=original, synthetic=synthetic,
                                                    original_name=self.original_name,
                                                    synthetic_name=self.synthetic_name)
                r = wasserstein.evaluate(metric=WassersteinMethod.SINKHORN, n_iterations=5, n_samples=500)
                utility_res[name] = r
                self.results.at[self.synthetic_name, "WASSERSTEIN"] = r
                self._save_txt(name, r, self.similarity_dir)
                print(r)

    def extras(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
        artifacts = {}
        if "CorrelationPearson" in self.metrics:
            if (synthetic is None) or (self.similarity_dir / "CorrelationPearson_result.txt").exists():
                print("CorrelationPearson result already exists or synthetic data not provided, skipping computation.")
                return artifacts
            print("  CorrelationPearson...")
            artifacts.update(
                self._correlation(original, synthetic, CorrelationMethod.PEARSON, "CorrelationPearson"))
        if "CorrelationSpearman" in self.metrics:
            if (synthetic is None) or (self.similarity_dir / "CorrelationSpearman_result.txt").exists():
                print("CorrelationSpearman result already exists or synthetic data not provided, skipping computation.")
                return artifacts
            print("  CorrelationSpearman...")
            artifacts.update(
                self._correlation(original, synthetic, CorrelationMethod.SPEARMAN, "CorrelationSpearman"))
        if "PCA" in self.metrics:
            if (synthetic is None) or (self.similarity_dir / "PCA_Original_vs_Synthetic.png").exists():
                print("PCA result already exists or synthetic data not provided, skipping computation.")
                return artifacts
            print("  PCA...")
            artifacts["PCA"] = str(self._pca(original, synthetic))
        if "PCA3D" in self.metrics:
            if (synthetic is None) or (self.similarity_dir / "PCA3D_Original_vs_Synthetic.html").exists():
                print("PCA3D result already exists or synthetic data not provided, skipping computation.")
                return artifacts
            print("  PCA3D...")
            artifacts["PCA3D"] = str(self._pca_3d(original, synthetic))
        if "Consensus" in self.metrics:
            if (self.similarity_dir / "Consensus_result.txt").exists():
                print("Consensus result already exists, skipping computation.")
                return artifacts
            print("  Consensus...")
            if not self.run_dir:
                raise ValueError(
                    "Consensus metric requires --run-dir pointing to the run folder that contains agent_* subdirs.")
            artifacts["Consensus"] = self._consensus(self.run_dir)

    def evaluate_iterations(self, gap: int = 100):
        """
        Evaluate privacy and statistical metrics at checkpoints
        Parameters
        ----------
        gap : int
            Interval between iterations to evaluate (e.g., every 50 iters)

        Returns
        -------

        """
        base_dir = self.run_dir / "agent_00"
        pattern = base_dir / "iter-*-model.pkl"
        files = sorted(glob.glob(str(pattern)))
        iterations = []

        for p in files:
            pth = Path(p)
            m = re.search(r"iter-(\d+)-model\.pkl$", pth.name)
            if not m:
                continue
            t = int(m.group(1))
            if t % gap != 0:
                continue
            iterations.append((t, pth))

        if not iterations:
            raise ValueError(f"No iteration checkpoints found under {base_dir}")

        print(f"Found {len(iterations)} checkpoints to evaluate (every {gap} iters).")

        iterations.sort(key=lambda x: x[0])
        it_root = self.output_dir / "iterations"
        it_root.mkdir(parents=True, exist_ok=True)

        all_privacy, all_utility = {}, {}
        metrics = [m for m in self.metrics if m.lower() != "consensus"]
        for t, pth in iterations:
            print(f"Iteration {t}...")
            out_dir = it_root / f"iter-{t:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            eval_i = Evaluator(
                manifest_path=self.manifest_path,
                model_path=str(pth),
                output_dir=str(out_dir),
                original_name=self.original_name,
                synthetic_name=self.synthetic_name,
                metrics=metrics,
                keys=self.keys,
                target=self.target,
                inf_aux_cols=self.inf_aux_cols,
                secret=self.secret,
                regression=self.regression,
                link_aux_cols=self.link_aux_cols,
                control_frac=self.control_frac,
                run_dir=None
            )
            eval_i.seed = self.seed
            res = eval_i.evaluate()
            all_privacy[t] = res["privacy"]
            all_utility[t] = res["utility"]
            print(f"  Privacy: {res['privacy']}")
            print(f"  Utility: {res['utility']}")
            p_priv = out_dir / "privacy.txt"
            p_util = out_dir / "utility.txt"
            with open(p_priv, "w", encoding="utf-8") as f:
                f.write(str(res["privacy"]))
            with open(p_util, "w", encoding="utf-8") as f:
                f.write(str(res["utility"]))

        return all_privacy, all_utility

    # --- helpers ---
    @staticmethod
    def _eval_privacy(metric):
        m = PrivacyMetricManager()
        m.add_metric(metric)
        return m.evaluate_all()

    @staticmethod
    def _eval_utility(metric):
        m = UtilityMetricManager()
        m.add_metric(metric)
        return m.evaluate_all()

    def _correlation(self, full_train, synthetic, method: CorrelationMethod, label: str) -> dict[str, str]:
        metric = CorrelationCalculator(full_train, synthetic, self.original_name, self.synthetic_name)
        score = metric.evaluate(method=method)
        with open(self.similarity_dir / f"{label}_result.txt", "w") as f:
            f.write(f"{label}: {score}\n")

        orig_df, syn_df = metric.correlation_pairs(method=method)
        p_orig_csv = self.similarity_dir / f"{label}_original_correlation.csv"
        p_syn_csv = self.similarity_dir / f"{label}_synthetic_correlation.csv"
        orig_df.to_csv(p_orig_csv, index=False)
        syn_df.to_csv(p_syn_csv, index=False)

        # quick heatmaps
        plt.figure(figsize=(9, 7))
        sns.heatmap(orig_df, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Original ({method.name.title()})")
        p1 = self.similarity_dir / f"{label}_original_heatmap.png"
        plt.savefig(p1)
        plt.clf()
        plt.figure(figsize=(9, 7))
        sns.heatmap(syn_df, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Synthetic ({method.name.title()})")
        p2 = self.similarity_dir / f"{label}_synthetic_heatmap.png"
        plt.savefig(p2)
        plt.clf()

        return {
            f"{label}_score": str(self.similarity_dir / f"{label}_result.txt"),
            f"{label}_orig_csv": str(p_orig_csv),
            f"{label}_syn_csv": str(p_syn_csv),
            f"{label}_orig_heatmap": str(p1),
            f"{label}_syn_heatmap": str(p2),
        }

    def _pca(self, full_train, synthetic) -> Path:
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
        out = self.similarity_dir / "PCA_Original_vs_Synthetic.png"
        plt.savefig(out)
        plt.clf()
        return out

    def _pca_3d(self, full_train: pd.DataFrame, synthetic: pd.DataFrame) -> Path:
        """
        Build an interactive 3D PCA scatter with Plotly and save as HTML.
        Returns the HTML path.
        """
        from sklearn.decomposition import PCA
        import plotly.graph_objs as go
        from plotly.offline import plot as plotly_plot

        combined = pd.concat([full_train, synthetic], ignore_index=True)
        X = combined.select_dtypes(include=[np.number]).fillna(0)

        pca = PCA(n_components=3)
        coords = pca.fit_transform(X)

        n = len(full_train)
        x_o, y_o, z_o = coords[:n, 0], coords[:n, 1], coords[:n, 2]
        x_s, y_s, z_s = coords[n:, 0], coords[n:, 1], coords[n:, 2]

        evr = pca.explained_variance_ratio_
        title = (f"PCA 3D: Original vs Synthetic "
                 f"(Explained Var %: PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%})")

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x_o, y=y_o, z=z_o,
            mode="markers",
            name="Original",
            opacity=0.6,
            marker=dict(size=3)
        ))
        fig.add_trace(go.Scatter3d(
            x=x_s, y=y_s, z=z_s,
            mode="markers",
            name="Synthetic",
            opacity=0.6,
            marker=dict(size=3)
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            legend=dict(x=0.01, y=0.99)
        )

        out = self.similarity_dir / "PCA3D_Original_vs_Synthetic.html"
        plotly_plot(fig, filename=str(out), auto_open=False, include_plotlyjs='cdn')
        return out

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

    def extract_run_info(self, run_dir: Path) -> str:
        # Example: run-20251017-112942-10Agents-30Epochs-1000Iterations-full-tabddpm
        name = run_dir.name
        match = re.search(
            r'run-\d+-\d+-(\d+)Agents-(\d+)Epochs-(\d+)Iterations-([^-]+)-([^-]+)', name
        )
        if not match:
            raise ValueError("Run directory name does not match expected pattern.")
        agents, epochs, iters, topology, model_type = match.groups()
        return f"{agents}A {epochs}E {iters}R {topology.capitalize()} {model_type.capitalize()}"

    def _consensus(self, run_dir: Path) -> dict[str, str]:
        """
        Run consensus metric, save artifacts (plots + CSVs) under output_dir/Consensus,
        and return a dict of artifact paths.
        """
        out_dir = self.output_dir / "Consensus"
        out_dir.mkdir(parents=True, exist_ok=True)
        consensus_run_dir= run_dir.parent
        run_name = self.extract_run_info(consensus_run_dir)
        # Per layer means
        self.plot_per_layer_consensus(consensus_run_dir, out_dir)
        res = self._consensus_curves_full(consensus_run_dir)
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
        plt.ylabel("Relative distance")
        plt.title(f"Per-agent relative distance to mean model parameters. {run_name}")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(fig1)
        plt.clf()

        fig1_linear = out_dir / "consensus-error-per-agent-linear.png"
        plt.figure()
        for ag in agents:
            plt.plot(iters, res["E_rel"][ag], label=ag, alpha=0.9)
        plt.xlabel("Round")
        plt.ylabel("Relative distance")
        plt.title(f"Per-agent relative distance to mean model parameters. {run_name}")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(fig1_linear)
        plt.clf()

        fig1_abs_linear = out_dir / "consensus-error-per-agent-abs-linear.png"
        plt.figure()
        for ag in agents:
            plt.plot(iters, res["E_abs"][ag], label=ag, alpha=0.9)
        plt.xlabel("Round")
        plt.ylabel("Distance")
        plt.title(f"Per-agent distance to mean model parameters. {run_name}")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(fig1_abs_linear)
        plt.clf()

        # 2) Average errors
        fig2 = out_dir / "consensus-error-average.png"
        plt.figure()
        # plt.plot(iters, avg_df["E_rel_mean"], label="Mean (rel)", linewidth=2)
        plt.plot(iters, avg_df["E_abs_mean"], label="Mean (abs)", linewidth=2)
        #plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Average distance")
        plt.title(f"Average distance to mean model parameters. {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig2)
        plt.clf()

        fig2 = out_dir / "consensus-error-average-rel.png"
        plt.figure()
        plt.plot(iters, avg_df["E_rel_mean"], label="Mean (rel)", linewidth=2)
        #plt.plot(iters, avg_df["E_abs_mean"], label="Mean (abs)", linewidth=2)
        # plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Average relative distance")
        plt.title(f"Average relative distance to mean model parameters. {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig2)
        plt.clf()

        # 3) Diameter
        fig3 = out_dir / "Max_pairwise_dist.png"
        plt.figure()
        plt.plot(iters, avg_df["D_rel"], label="Normalized", linewidth=2)
        plt.plot(iters, avg_df["D_abs"], label="Absolute", linestyle="--", linewidth=2)
        #plt.yscale("log")
        plt.xlabel("Round")
        plt.ylabel("Max pairwise distance")
        plt.title(f"Max pairwise distance between agents model parameters. {run_name}")
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
        plt.title(f"Total squared distance to mean model parameters. {run_name}")
        plt.ylabel("Sum of squared distances to mean parameters")
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

    def plot_per_layer_consensus(self, run_dir: Path, output_dir: Path):
        # Find all agent dirs and iterations
        agent_dirs = sorted(run_dir.glob("agent_*"))
        per_layer = defaultdict(lambda: defaultdict(list))  # layer -> agent -> [means]

        # Collect all iterations
        iter_nums = None
        for agent_dir in agent_dirs:
            agent = agent_dir.name
            model_files = sorted(agent_dir.glob("iter-*-weights.pt"))
            agent_iters = []
            for mf in model_files:
                m = re.search(r"iter-(\d+)-weights\.pt$", mf.name)
                if m:
                    agent_iters.append((int(m.group(1)), mf))
            agent_iters.sort()
            if iter_nums is None:
                iter_nums = [t for t, _ in agent_iters]
            # For each iteration, compute per-layer means
            for t, mf in agent_iters:
                state = torch.load(mf, map_location="cpu")
                means = per_layer_means(state)
                flat_means = _flatten_means(means)
                for layer, mean in flat_means.items():
                    per_layer[layer][agent].append(mean)

        # Plot for each layer
        output_dir.mkdir(parents=True, exist_ok=True)
        for layer, agent_means in per_layer.items():
            plt.figure()
            for agent, means in agent_means.items():
                plt.plot(iter_nums, means, label=agent)
            plt.xlabel("Iteration")
            plt.ylabel("Mean")
            plt.title(f"Consensus: Per-layer mean for {layer}")
            plt.legend(fontsize="small")
            plt.tight_layout()
            out_path = output_dir / f"consensus_per_layer_{layer.replace('.', '_')}.png"
            plt.savefig(out_path)
            plt.close()

    def per_layer_means(self, model_path=None):
        """Compute per-layer means for a model."""
        model_path = model_path or self.model_path
        state = torch.load(model_path, map_location="cpu")
        means = per_layer_means(state)
        flat_means = _flatten_means(means)
        # Save as CSV
        out_path = self.output_dir / "per_layer_means.csv"
        import pandas as pd
        pd.Series(flat_means).to_csv(out_path)
        print(f"Per-layer means saved to {out_path}")
        return flat_means

    def _save_txt(self, metric_name: str, result,  base_dir: Path = None) -> None:
        base = base_dir if base_dir is not None else self.output_dir
        base.mkdir(parents=True, exist_ok=True)
        p = base / f"{metric_name}_result.txt"
        with open(p, "w", encoding="utf-8") as f:
            if isinstance(result, dict):
                for k, v in result.items(): f.write(f"{k}: {v}\n")
            else:
                f.write(f"value: {result}\n")

def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")] if s else []

def cli(argv: list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data privacy & utility metrics."
    )

    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing original data CSV (train.csv)."
    )
    parser.add_argument(
        "--categorical-cols", type=_csv_list, default=[],
        help="Comma-separated list of categorical column names."
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained generative model file (pickle)."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--metrics", type=_csv_list, default=None,
        help="Comma-separated list of metrics to compute. "
    )
    parser.add_argument(
        "--keys", type=_csv_list, default=None,
        help="Comma-separated list of key columns for Disclosure metric."
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target column for Disclosure metric."
    )
    parser.add_argument(
        "--inf-aux-cols", type=_csv_list, default=None,
        help="Comma-separated list of auxiliary columns for Inference metric."
    )
    parser.add_argument(
        "--secret", type=str, default=None,
        help="Secret column for Inference metric."
    )
    parser.add_argument(
        "--regression", action="store_true",
        help="Flag indicating if the secret is continuous (regression). Default is classification."
    )
    parser.add_argument(
        "--link-aux-cols", type=str, default=None,
        help="Comma-separated pair of auxiliary column lists for Linkability metric, e.g., 'col1,col2;col3,col4'."
    )
    parser.add_argument(
        "--control-frac", type=float, default=0.3,
        help="Fraction of original data to use as control for Linkability metric. Default is 0.3."
    )
    parser.add_argument(
        "--original-name", type=str, default="adult",
        help="Name of the original dataset (for labeling). Default is 'adult'."
    )
    parser.add_argument(
        "--synthetic-name", type=str, default="CTGAN",
        help="Name of the synthetic dataset/model (for labeling). Default is 'CTGAN'."
    )
    parser.add_argument(
        "--model-type", type=str, default="ctgan",
        help="Type of generative model (ctgan, tabddpm, etc.). Default is 'ctgan'."
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to the run directory containing agent subdirectories (for Consensus metric)."
    )
    parser.add_argument(
        "--eval-iters", action="store_true",
        help="If set, evaluate metrics at model iteration checkpoints under run-dir/agent_00."
    )
    parser.add_argument(
        "--iter-gap", type=int, default=50,
        help="Interval between iterations to evaluate if --eval-iters is set. Default is 50."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default is 42."
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="If set, run baseline evaluation (no synthetic data) for Consensus metric."
    )

    args = parser.parse_args(argv)

    orig_metrics = args.metrics
    selected = set([m.lower() for m in args.metrics]) if args.metrics else set()

    def want(name: str) -> bool:
        return name.lower() in selected

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

    if args.eval_iters:
        if not args.run_dir:
            print("ERROR: --run-dir is required for --iter-eval (to locate agent_00/iter-*-model.pkl).",
                  file=sys.stderr)
            return 2

        print(f"[iter-eval] Running iteration sweep every {args.iter_interval} iters...")

        # Remove consensus from metrics if present
        if "consensus" in selected:
            selected.remove("consensus")
            orig_metrics = [m for m in orig_metrics if m.lower() != "consensus"]
            print("  Note: Consensus metric is skipped during iteration evaluation.")

        evaluator = Evaluator(
            data_dir=args.data_dir,
            categorical_cols=args.categorical_cols,
            model_path=args.model_path,
            output_dir=args.output_dir,
            original_name=args.original_name,
            synthetic_name=args.synthetic_name,
            model_type=args.model_type,
            metrics=orig_metrics,
            keys=args.keys,
            target=args.target,
            inf_aux_cols=args.inf_aux_cols,
            secret=args.secret,
            regression=args.regression,
            link_aux_cols=link_aux_cols,
            control_frac=args.control_frac,
            run_dir=None,  # No consensus during iter eval
        )
        evaluator.seed = args.seed

        all_privacy, all_utility = evaluator.evaluate_iterations(gap=args.iter_gap)
        p_priv = Path(args.output_dir) / "all_privacy_iterations.csv"
        p_util = Path(args.output_dir) / "all_utility_iterations.csv"
        df_priv = pd.DataFrame.from_dict(all_privacy, orient="index").sort_index
        df_util = pd.DataFrame.from_dict(all_utility, orient="index").sort_index
        df_priv.to_csv(p_priv)
        df_util.to_csv(p_util)
        return 0

    if args.baseline:
        print("Running baseline evaluation (no synthetic data)...")
        if not args.run_dir:
            print("ERROR: --run-dir is required when running baseline (for Consensus metric).", file=sys.stderr)
            return 2

        evaluator = Evaluator(
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            categorical_cols=args.categorical_cols,
            model_path=args.model_path,
            output_dir=args.output_dir,
            original_name=args.original_name,
            synthetic_name=args.synthetic_name,
            model_type=args.model_type,
            metrics=orig_metrics,
            keys=args.keys,
            target=args.target,
            inf_aux_cols=args.inf_aux_cols,
            secret=args.secret,
            regression=args.regression,
            link_aux_cols=link_aux_cols,
            control_frac=args.control_frac,
        )
        evaluator.seed = args.seed
        evaluator.evaluate()
        return 0

    # loop over agents
    run_dir = Path(args.run_dir)
    agent_dirs = sorted([p for p in run_dir.glob("agent_*") if p.is_dir()])

    if not agent_dirs:
        print("ERROR: No agents found in run-dir", file=sys.stderr)
        return 2

    for agent_dir in agent_dirs:
        model_path = agent_dir / args.model_path
        if not model_path.exists():
            print(f"ERROR: Model {model_path} not found in agent dir {agent_dir}.", file=sys.stderr)
            return 2

        print(f"Evaluating agent dir: {agent_dir} with model: {model_path}")
        out_dir = agent_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        evaluator = Evaluator(
            data_dir=args.data_dir,
            categorical_cols=args.categorical_cols,
            model_path=str(model_path),
            output_dir=str(out_dir),
            original_name=args.original_name,
            synthetic_name=f"{args.synthetic_name}_{agent_dir.name}",
            model_type=args.model_type,
            metrics=orig_metrics,
            keys=args.keys,
            target=args.target,
            inf_aux_cols=args.inf_aux_cols,
            secret=args.secret,
            regression=args.regression,
            link_aux_cols=link_aux_cols,
            control_frac=args.control_frac,
            run_dir=args.run_dir
        )
        evaluator.seed = args.seed

        results = evaluator.evaluate()
        print("\n========= SUMMARY =========\n")
        print(results)
    return 0

if __name__ == "__main__":
    sys.exit(cli())