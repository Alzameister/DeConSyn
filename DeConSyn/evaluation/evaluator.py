from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from DeConSyn.data.data_loader import DatasetLoader, ADULT_PATH, ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET
from DeConSyn.evaluation.consensus import consensus
from DeConSyn.io.io import get_config_dir, load_model_pickle
from DeConSyn.models.tab_ddpm.lib import load_config
from DeConSyn.models.tab_ddpm.scripts.sample import sample
from DeFeSyn.utils.seed import set_global_seed
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
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.js_similarity import \
    JSCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import \
    KSCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.wasserstein import \
    WassersteinCalculator


class Evaluator:
    def __init__(
            self,
            original_data: pd.DataFrame,
            original_data_path: str,
            categorical_columns: list[str],
            agent_dir: str,
            metrics: list[str],
            model_type: str = "ctgan",
            model_name: str = "ctgan.pkl",
            dataset_name: str = "adult",
            synthetic_name: str = "CTGAN",
            keys: list[str] = None,
            target: str = None,
            seed: int = 42,
            iteration: int = None,
            baseline: bool = False,
            baseline_dir: str = None
    ):
        self.original_data: pd.DataFrame = original_data
        self.data_dir: Path = Path(original_data_path)
        self.categorical_columns: list[str] = categorical_columns

        self.model_type: str = model_type
        self.model_name: str = model_name
        self.baseline = baseline
        self.baseline_dir = baseline_dir

        self.dataset_name: str = dataset_name
        self.synthetic_name: str = synthetic_name

        self.metrics: list[str] = metrics
        self.keys: list[str] = keys
        self.target: str = target
        self.seed: int = seed

        self.run_dir: Path = Path(agent_dir)
        self.model_path: Path = self.run_dir / self.model_name
        self.results: pd.DataFrame = pd.DataFrame(columns=[
            "DCR", "NNDR", "AdversarialAccuracy",
            "RepU", "Disclosure",
            "Mean", "Median", "Var", "JS", "KS", "WASSERSTEIN",
            "CorrelationPearson", "CorrelationSpearman", "PCA",
            "Consensus"
        ], index=[self.synthetic_name])
        self.privacy_metrics = ["DCR", "NNDR", "AdversarialAccuracy", "Disclosure",
            "RepU", "DiSCO"]
        self.similarity_metrics = ["Mean", "Median", "Var", "JS", "KS", "WASSERSTEIN",
            "CorrelationPearson", "CorrelationSpearman", "PCA"]
        if iteration is not None:
            self.results_dir = self.run_dir / f"results-iter-{iteration:05d}"
        else:
            self.results_dir: Path = self.run_dir / "results"
        self.results_file: Path = self.results_dir / "results.csv"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.validate_requirements()

        print("Using model path:", self.model_path)

    def validate_requirements(self):
        # Validate that required parameters are set for specific metrics
        if "Disclosure" in self.metrics and (self.keys is None or self.target is None):
            raise ValueError("Keys and target must be provided for Disclosure metric.")

    def evaluate(self) -> pd.DataFrame:
        self.get_calculated_metrics()

        # If all metrics already covered, return results
        if self.all_metrics_covered():
            return self.results

        if not self.metrics == ['Consensus']:
            synthetic: pd.DataFrame = self.get_synthetic()
            self.plot_distributions(self.original_data, synthetic)
            self.calculate_privacy_metrics(self.original_data, synthetic)
            self.calculate_similarity_metrics(self.original_data, synthetic)

        if 'Consensus' in self.metrics:
            consensus(self.run_dir.parent)

        if self.baseline:
            synthetic.to_csv(self.results_dir / f"{self.synthetic_name}_synthetic.csv", index=False)


        self.results.to_csv(self.results_file, index=True)
        return self.results

    def get_result_columns_for_metric(self, metric: str) -> list[str]:
        """Map a metric name to the result columns it produces."""
        metric_to_columns = {
            "Correlation": ["CorrelationPearson", "CorrelationSpearman"],
            "BasicStats": ["Mean", "Median", "Var"],
            "DCR": ["DCR"],
            "NNDR": ["NNDR"],
            "AdversarialAccuracy": ["AdversarialAccuracy"],
            "RepU": ["RepU"],
            "Disclosure": ["Disclosure"],
            "JS": ["JS"],
            "KS": ["KS"],
            "WASSERSTEIN": ["WASSERSTEIN"],
        }
        return metric_to_columns.get(metric, [metric])

    def all_metrics_covered(self) -> bool:
        """Check if all required result columns for the selected metrics are present and not NaN."""
        for metric in self.metrics:
            if metric in ['Consensus', 'PCA']:
                continue
            for col in self.get_result_columns_for_metric(metric):
                if col not in self.results.columns or pd.isna(self.results.at[self.synthetic_name, col]):
                    return False
        return True

    def get_calculated_metrics(self):
        """Checks if metrics have been calculated and saved in a file --> Append to results."""
        if self.results_file.exists():
            saved_results = pd.read_csv(self.results_file, index_col=0)
            for metric in self.metrics:
                for col in self.get_result_columns_for_metric(metric):
                    if col in saved_results.columns and pd.notna(saved_results.at[self.synthetic_name, col]):
                        print(f"Loading saved metric column: {col}")
                        self.results.at[self.synthetic_name, col] = saved_results.at[self.synthetic_name, col]

    def get_synthetic(self) -> pd.DataFrame:
        if self.model_type == "ctgan":
            return self.load_ctgan()
        elif self.model_type == "tabddpm":
            return self.load_tabddpm()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def load_tabddpm(self) -> pd.DataFrame:
        int_cols = self.original_data.select_dtypes(include=['int64']).columns
        self.original_data[int_cols] = self.original_data[int_cols].astype('float64')
        config = load_config(get_config_dir() / self.dataset_name / "config.toml")
        set_global_seed(self.seed)

        sample(
            parent_dir=self.run_dir,
            model_path=self.model_path,
            real_data_path=str(self.data_dir) + "/npy",
            num_samples=len(self.original_data),
            batch_size=10000,
            disbalance=config['sample'].get('disbalance', None),
            **config['diffusion_params'],
            model_type=config['model_type'],
            model_params=config['model_params'],
            T_dict=config['train']['T'],
            num_numerical_features=config['num_numerical_features'],
            device="cpu",
            seed=self.seed,
            change_val=False
        )

        x_cat_p = self.run_dir / 'X_cat_train.npy'
        x_num_p = self.run_dir / 'X_num_train.npy'
        y_p = self.run_dir / 'y_train.npy'

        x_cat = np.load(x_cat_p, allow_pickle=True)
        x_num = np.load(x_num_p, allow_pickle=True)
        y = np.load(y_p, allow_pickle=True)

        x_gen = np.concatenate([x_num, x_cat], axis=1)
        synthetic = pd.DataFrame(x_gen, columns=self.original_data.columns.drop(self.target))
        synthetic[self.target] = np.asarray(y).squeeze()

        # Ensure categorical columns have the same dtype as original
        for col in self.categorical_columns:
            synthetic[col] = synthetic[col].astype(self.original_data[col].dtype)

        # Ensure numerical are float
        for col in self.original_data.columns:
            if col not in self.categorical_columns and col != self.target:
                synthetic[col] = synthetic[col].astype('float64')

        return synthetic

    def load_ctgan(self) -> pd.DataFrame:
        model = load_model_pickle(Path(self.model_path))
        set_global_seed(self.seed)
        return model.sample(len(self.original_data))
        # return model.sample(10_000)

    def calculate_privacy_metrics(self, original: pd.DataFrame, synthetic: pd.DataFrame):
        print("Calculating privacy metrics...")
        # Only go through self.metrics that are in privacy_metrics
        for metric in self.metrics:
            if metric in self.privacy_metrics and pd.isna(self.results.at[self.synthetic_name, metric]):
                print(f"Calculating {metric}...")
                if metric == "DCR":
                    dcr_calculator = DCRCalculator(original=original, synthetic=synthetic,
                                                   original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    dcr_value = dcr_calculator.evaluate()
                    self.results.at[self.synthetic_name, "DCR"] = dcr_value
                if metric == "NNDR":
                    nndr_calculator = NNDRCalculator(original=original, synthetic=synthetic,
                                                     original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    nndr_value = nndr_calculator.evaluate()
                    self.results.at[self.synthetic_name, "NNDR"] = nndr_value
                if metric == "AdversarialAccuracy":
                    aa_calculator = AdversarialAccuracyCalculator(original=original, synthetic=synthetic,
                                                                  original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    aa_value = aa_calculator.evaluate()
                    self.results.at[self.synthetic_name, "AdversarialAccuracy"] = aa_value
                if metric == "Disclosure":
                    disco_calculator = DisclosureCalculator(original=original, synthetic=synthetic,
                                                            original_name=self.dataset_name, synthetic_name=self.synthetic_name,
                                                            keys=self.keys, target=self.target)
                    repu_value, disco_value = disco_calculator.evaluate()
                    self.results.at[self.synthetic_name, "RepU"] = repu_value
                    self.results.at[self.synthetic_name, "Disclosure"] = disco_value

        print("Calculated Privacy Metrics:")
        for metric in self.privacy_metrics:
            if metric in self.metrics:
                value = self.results.at[self.synthetic_name, metric]
                print(f"{metric}: {value}")

    def calculate_similarity_metrics(self, original: pd.DataFrame, synthetic: pd.DataFrame):
        print("Calculating similarity metrics...")
        # Only go through self.metrics that are in similarity_metrics
        for metric in self.metrics:
            print(f"Calculating {metric}...")
            if metric == "BasicStats":
                if pd.isna(self.results.at[self.synthetic_name, "Mean"]):
                    basicstats_calculator = BasicStatsCalculator(original=original, synthetic=synthetic,
                                                                 original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    stats_results = basicstats_calculator.evaluate()
                    for stat_name, stat_value in stats_results.items():
                        self.results.at[self.synthetic_name, stat_name.capitalize()] = stat_value
            elif metric == "Correlation" and pd.isna(self.results.at[self.synthetic_name, "CorrelationPearson"]):
                self.calculate_correlation(original, synthetic)
            elif metric in self.similarity_metrics and pd.isna(self.results.at[self.synthetic_name, metric]):
                if metric == "JS":
                    js_calculator = JSCalculator(original=original, synthetic=synthetic,
                                                 original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    js_value = js_calculator.evaluate()
                    self.results.at[self.synthetic_name, "JS"] = js_value
                if metric == "KS":
                    ks_calculator = KSCalculator(original=original, synthetic=synthetic,
                                                 original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    ks_value = ks_calculator.evaluate()
                    self.results.at[self.synthetic_name, "KS"] = ks_value
                if metric == "WASSERSTEIN":
                    wasserstein_calculator = WassersteinCalculator(original=original, synthetic=synthetic,
                                                                   original_name=self.dataset_name, synthetic_name=self.synthetic_name)
                    # metric=WassersteinMethod.SINKHORN, n_iterations=5, n_samples=500 maybe
                    wasserstein_value = wasserstein_calculator.evaluate()
                    self.results.at[self.synthetic_name, "WASSERSTEIN"] = wasserstein_value
                if metric == "PCA":
                    self.calculate_pca(original, synthetic)


        print("Calculated Similarity Metrics:")
        for metric in self.similarity_metrics:
            if metric in self.metrics:
                value = self.results.at[self.synthetic_name, metric]
                print(f"{metric}: {value}")

    def calculate_correlation(self, original: pd.DataFrame, synthetic: pd.DataFrame):
        output_dir = self.results_dir / 'Correlation'
        output_dir.mkdir(parents=True, exist_ok=True)
        original_numeric = original.select_dtypes(include=[np.number])
        synthetic_numeric = synthetic.select_dtypes(include=[np.number])
        # Pearson
        pearson_corr = original_numeric.corr(method='pearson')
        synthetic_pearson_corr = synthetic_numeric.corr(method='pearson')
        pearson_diff = np.abs(pearson_corr - synthetic_pearson_corr)
        pearson_mean_diff = np.mean(pearson_diff)
        self.results.at[self.synthetic_name, "CorrelationPearson"] = pearson_mean_diff

        # Save original, synthetic and heatmap of differences
        pearson_corr.to_csv(output_dir / 'original_pearson_corr.csv')
        synthetic_pearson_corr.to_csv(output_dir / 'synthetic_pearson_corr.csv')
        pearson_diff.to_csv(output_dir / 'pearson_correlation_difference.csv')
        diff_plot_path = output_dir / 'pearson_correlation_difference_heatmap.png'
        plt.figure(figsize=(12, 10))
        sns.heatmap(pearson_diff, annot=True, fmt=".2f", cmap='viridis')
        plt.title('Pearson Correlation Difference Heatmap')
        plt.savefig(diff_plot_path)
        plt.clf()

        # Spearman
        spearman_corr = original_numeric.corr(method='spearman')
        synthetic_spearman_corr = synthetic_numeric.corr(method='spearman')
        spearman_diff = np.abs(spearman_corr - synthetic_spearman_corr)
        spearman_mean_diff = np.mean(spearman_diff)
        self.results.at[self.synthetic_name, "CorrelationSpearman"] = spearman_mean_diff

        # Save original, synthetic and heatmap of differences
        spearman_corr.to_csv(output_dir / 'original_spearman_corr.csv')
        synthetic_spearman_corr.to_csv(output_dir / 'synthetic_spearman_corr.csv')
        spearman_diff.to_csv(output_dir / 'spearman_correlation_difference.csv')
        diff_plot_path = output_dir / 'spearman_correlation_difference_heatmap.png'
        plt.figure(figsize=(12, 10))
        sns.heatmap(spearman_diff, annot=True, fmt=".2f", cmap='viridis')
        plt.title('Spearman Correlation Difference Heatmap')
        plt.savefig(diff_plot_path)
        plt.clf()

    def calculate_pca(self, original: pd.DataFrame, synthetic: pd.DataFrame):
        output_dir = self.results_dir / 'PCA'
        output_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([original, synthetic], ignore_index=True)
        combined = pd.get_dummies(combined, columns=self.categorical_columns)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        n = len(original)

        plt.figure(figsize=(12, 8))
        plt.scatter(pca_result[:n, 0], pca_result[:n, 1], label='Original', alpha=0.5)
        plt.scatter(pca_result[n:, 0], pca_result[n:, 1], label='Synthetic', alpha=0.5)
        plt.title('PCA of Original and Synthetic Data')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        pca_plot_path = output_dir / 'pca_plot.png'
        plt.savefig(pca_plot_path)
        plt.clf()

    def plot_distributions(self, original: pd.DataFrame, synthetic: pd.DataFrame):
        if not self.baseline and self.baseline_dir is not None:
            baseline_synthetic_path = Path(self.baseline_dir) / "results" / f"{self.dataset_name}_synthetic.csv"
            if baseline_synthetic_path.exists():
                baseline_synthetic = pd.read_csv(baseline_synthetic_path)
                self._plot_distributions_with_baseline(original, synthetic, baseline_synthetic)

        output_dir = self.results_dir / 'Distributions'
        output_dir.mkdir(parents=True, exist_ok=True)
        columns = original.columns
        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, column in enumerate(columns):
            ax = axes[idx]
            if column in self.categorical_columns:
                orig_freq = original[column].value_counts(normalize=True).sort_index()
                synth_freq = synthetic[column].value_counts(normalize=True).sort_index()
                all_cats = sorted(set(orig_freq.index).union(synth_freq.index))
                orig_freq = orig_freq.reindex(all_cats, fill_value=0)
                synth_freq = synth_freq.reindex(all_cats, fill_value=0)
                width = 0.4
                x = np.arange(len(all_cats))
                ax.bar(x - width / 2, orig_freq, width=width, color='blue', alpha=0.7, label='Original')
                ax.bar(x + width / 2, synth_freq, width=width, color='orange', alpha=0.7, label='Synthetic')
                ax.set_xticks(x)
                ax.set_xticklabels(all_cats, rotation=45, ha='right')
                ax.set_ylabel('Proportion')
            else:
                sns.histplot(original[column], color='blue', label='Original', stat='probability', ax=ax, alpha=0.3)
                sns.histplot(synthetic[column], color='orange', label='Synthetic', stat='probability', ax=ax, alpha=0.3)
                # sns.kdeplot(original[column], color='blue', ax=ax)
                # sns.kdeplot(synthetic[column], color='orange', ax=ax)
                ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.legend()

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plot_path = output_dir / 'all_distributions.png'
        plt.savefig(plot_path)
        plt.clf()

    def _plot_distributions_with_baseline(self, original: pd.DataFrame, synthetic: pd.DataFrame, baseline_synthetic: pd.DataFrame):
        output_dir = self.results_dir / 'Distributions'
        output_dir.mkdir(parents=True, exist_ok=True)
        columns = original.columns
        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, column in enumerate(columns):
            ax = axes[idx]
            if column in self.categorical_columns:
                synth_freq = synthetic[column].value_counts(normalize=True).sort_index()
                baseline_freq = baseline_synthetic[column].value_counts(normalize=True).sort_index()
                all_cats = sorted(set(synth_freq.index).union(baseline_freq.index))
                synth_freq = synth_freq.reindex(all_cats, fill_value=0)
                baseline_freq = baseline_freq.reindex(all_cats, fill_value=0)
                x = np.arange(len(all_cats))
                ax.bar(x + 0.2, baseline_freq, width=0.2, color='blue', alpha=0.5, label='Baseline')
                ax.bar(x, synth_freq, width=0.2, color='orange', alpha=0.5, label='Synthetic')
                ax.set_xticks(x)
                ax.set_xticklabels(all_cats, rotation=45, ha='right')
                ax.set_ylabel('Proportion')
            else:
                sns.histplot(baseline_synthetic[column], color='blue', label='Baseline', stat='probability',
                             ax=ax, alpha=0.3)
                sns.histplot(synthetic[column], color='orange', label='Synthetic', stat='probability', ax=ax, alpha=0.3)

                #sns.kdeplot(original[column], color='blue', ax=ax)
                #sns.kdeplot(synthetic[column], color='orange', ax=ax)
                #sns.kdeplot(baseline_synthetic[column], color='green', ax=ax)
                ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.legend()

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plot_path = output_dir / 'all_distributions_with_baseline.png'
        plt.savefig(plot_path)
        plt.clf()


if __name__ == "__main__":
    loader = DatasetLoader(ADULT_PATH, ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET)
    original_data = loader.get_train()
    model_type = "ctgan"
    model_name = "iter-00300-model.pkl"
    run_dir = "/runs/ctgan/4A-1E-500R-Full/run-20251024-150720-4Agents-1Epochs-300Iterations-full-ctgan/agent_00"
    metrics = ['PCA', 'Consensus', 'Correlation']
    evaluator = Evaluator(original_data=original_data, original_data_path=ADULT_PATH,
                          categorical_columns=ADULT_CATEGORICAL_COLUMNS + [ADULT_TARGET], agent_dir=run_dir,
                          metrics=metrics, model_type=model_type, model_name=model_name, dataset_name="adult",
                          synthetic_name="CTGAN")
    results = evaluator.evaluate()