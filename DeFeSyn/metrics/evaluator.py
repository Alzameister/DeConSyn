import argparse
import sys
from pathlib import Path
from typing import List, Dict, Union, Tuple

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
        """
    def __init__(self, manifest_path: str, model_path: str, output_dir: str,
                 metrics: List[str] = None, keys: List[str] = None, target: str = None,
                 inf_aux_cols: List[str] = None, secret: str = None, regression: bool = False,
                 link_aux_cols: Tuple[List[str], List[str]] = None, control_frac: float = 0.3,
                 original_name: str = "adult", synthetic_name: str = "CTGAN"):
        self.seed = 42
        self.manifest_path = manifest_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_name = original_name
        self.synthetic_name = synthetic_name

        self.metrics = metrics if metrics is not None else [
            "DCR", "NNDR", "AdversarialAccuracy", "SinglingOut", "Inference", "Linkability", "Disclosure",
            "BasicStats", "JS", "KS",
            "CorrelationPearson", "CorrelationSpearman", "PCA"
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

        # ---- Privacy
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


        # ---- Utility
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

        # ---- Extras
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
                 "CorrelationPearson", "CorrelationSpearman", "PCA"],
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
    )

    # Optional: set seed if you also want to seed numpy/py modules consistently inside evaluate()
    evaluator.seed = args.seed

    results = evaluator.evaluate()
    print("\n========= SUMMARY =========")
    print(results)
    return 0

if __name__ == "__main__":
    raise SystemExit(cli())

# if __name__ == "__main__":
#     path = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/baseline_ctgan"
#     model_path = path + "/ctgan_adult_default.pkl"
#
#     #metrics = ["AdversarialAccuracy", "SinglingOut", "Inference", "Linkability", "Disclosure"]
#
#     # Disclosure settings
#     keys = ["age", "sex", "marital-status", "education", "occupation", "hours-per-week", "workclass",
#                  "native-country"]
#     target = "income"
#     # Inferential attack settings
#     inf_aux_cols = ['age', 'sex', 'race', 'relationship', 'education', 'occupation', 'workclass', 'native-country']
#     secret = 'income'
#     regression = False
#
#     # Linkability settings
#     link_aux_cols = (
#         ["age", "sex", "race", "marital-status", "native-country"],  # set A
#         ["education", "workclass", "occupation", "hours-per-week"]  # set B
#     )
#
#     evaluator = SynEvaluator(
#         manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}",
#         model_path=model_path,
#         output_dir=path,
#         original_name="adult",
#         synthetic_name="CTGAN 4 Agents 15 Epochs 20 Iterations Ring",
#         #metrics=metrics,
#         keys=keys,
#         target=target,
#         inf_aux_cols=inf_aux_cols,
#         secret=secret,
#         regression=regression,
#         link_aux_cols=link_aux_cols,
#         control_frac=0.3,
#     )
#     results = evaluator.evaluate()
#     print(results)
