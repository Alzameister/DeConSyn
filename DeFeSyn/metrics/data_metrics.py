from pathlib import Path

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.metrics.FEST.metrics.privacy_metrics.attacks.singlingout_class import \
    SinglingOutCalculator
from DeFeSyn.metrics.FEST.metrics.privacy_metrics.distance.adversarial_accuracy_class import \
    AdversarialAccuracyCalculator
from DeFeSyn.metrics.FEST.metrics.privacy_metrics.distance.dcr_class import DCRCalculator
from DeFeSyn.metrics.FEST.metrics.privacy_metrics.distance.nndr_class import NNDRCalculator
from DeFeSyn.metrics.FEST.metrics.privacy_metrics.privacy_metric_manager import \
    PrivacyMetricManager
from DeFeSyn.metrics.FEST.metrics.utility_metrics.statistical.basic_stats import \
    BasicStatsCalculator
from DeFeSyn.metrics.FEST.metrics.utility_metrics.statistical.correlation import \
    CorrelationMethod, CorrelationCalculator
from DeFeSyn.metrics.FEST.metrics.utility_metrics.statistical.js_similarity import \
    JSCalculator
from DeFeSyn.metrics.FEST.metrics.utility_metrics.statistical.ks_test import KSCalculator
from DeFeSyn.metrics.FEST.metrics.utility_metrics.utility_metric_manager import \
    UtilityMetricManager
from DeFeSyn.spade_model.start import ADULT_MANIFEST, ADULT_PATH
from DeFeSyn.utils.io import load_model_pickle

if __name__ == "__main__":
    path = Path("C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/run-20250831-171833-a4-e15-i20-alpha1.0-ring")
    # path = Path("C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/baseline_ctgan")
    manifest_path = f"{ADULT_PATH}/{ADULT_MANIFEST}"
    loader = DatasetLoader(manifest_path)
    full_train = loader.get_train()
    full_test = loader.get_test()

    model = load_model_pickle(path / "agent_00/iter-00020-model.pkl")
    # model = load_model_pickle(path / "ctgan_adult_default.pkl")
    synthetic_data = model.sample(len(full_train))
    # full_train = full_train.sample(len(full_train), random_state=42).reset_index(drop=True)

    # Remove missing values if any
    synthetic_data = synthetic_data.dropna()
    full_train = full_train.dropna()

    # Remove rows that have "workclass" == "Never-worked"
    synthetic_data = synthetic_data[synthetic_data["workclass"] != "Never-worked"]
    full_train = full_train[full_train["workclass"] != "Never-worked"]

    original_name = "adult"
    synthetic_name = "CTGAN"

    # Disco settings
    keys = ["age", "sex", "marital-status", "education", "occupation", "hours-per-week", "workclass", "native-country"]
    target = "income"

    # Inferential attack settings
    inf_aux_cols = ['age', 'sex', 'race', 'relationship', 'education', 'occupation', 'workclass', 'native-country']
    secret = 'income'
    regression = False

    # Linkability settings
    link_aux_cols = (
        ["age", "sex", "race", "marital-status", "native-country"],  # set A
        ["education", "workclass", "occupation", "hours-per-week"]  # set B
    )
    control_frac = 0.3
    link_control_df = full_train.sample(frac=control_frac, random_state=42)
    link_original_train_df = full_train.drop(link_control_df.index).reset_index(drop=True)
    link_control_df = link_control_df.reset_index(drop=True)

    metric_list = [
        # TODO: DCR Buggy
        DCRCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        NNDRCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        AdversarialAccuracyCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name,
                                      synthetic_name=synthetic_name),
        # TODO: Disc Error: raise ValueError("keys and target must be variables in data and synthetic data.")
        # DisclosureCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name,
        #                      keys=keys, target=target),
        SinglingOutCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name),
        # TODO: Inf Error: Use Categorical Encoder?
        # InferenceCalculator(original=full_train, synthetic=synthetic_data, aux_cols=inf_aux_cols, secret=secret,
        #                      regression=regression, original_name=original_name, synthetic_name=synthetic_name),
        # TODO: Link Error: Use Categorical Encoder?
        # LinkabilityCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name,
        #                      aux_cols=link_aux_cols, control=link_control_df)
    ]

    for metric in metric_list:
        print(f"Evaluating {metric.__class__.__name__}...")
        p = PrivacyMetricManager()
        p.add_metric(metric)
        result = p.evaluate_all()

        # Save individual metric results to separate files
        with open(path / f"{metric.__class__.__name__}_result.txt", "w") as f:
            for key, value in result.items():
                print(f"{key}: {value}")
                f.write(f"{key}: {value}\n")

    metric_list = [
        BasicStatsCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        JSCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        KSCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        # MICalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        # TODO: Allocate more memory
        # WassersteinCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
    ]

    for metric in metric_list:
        print(f"Evaluating {metric.__class__.__name__}...")
        p = UtilityMetricManager()
        p.add_metric(metric)
        result = p.evaluate_all()
        # Save individual metric results to separate files
        with open(path / f"{metric.__class__.__name__}_result.txt", "w") as f:
            for key, value in result.items():
                print(f"{key}: {value}")
                f.write(f"{key}: {value}\n")

    # Correlation Metric with detailed output
    metric = CorrelationCalculator(original=full_train, synthetic=synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
    score = metric.evaluate()
    correlation_pairs = metric.correlation_pairs(method=CorrelationMethod.PEARSON)
    orig_corr, syn_corr = correlation_pairs
    orig_corr.to_csv(path / "CorrelationCalculator_Pearson_original_correlation.csv", index=False)
    syn_corr.to_csv(path / "CorrelationCalculator_Pearson_synthetic_correlation.csv", index=False)
    # Create Heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sns.heatmap(orig_corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Original Data Correlation Heatmap (Pearson)")
    plt.savefig(path / "CorrelationCalculator_Pearson_original_correlation_heatmap.png")
    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.heatmap(syn_corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Synthetic Data Correlation Heatmap (Pearson)")
    plt.savefig(path / "CorrelationCalculator_Pearson_synthetic_correlation_heatmap.png")
    plt.clf()
    with open(path / f"CorrelationCalculator_Pearson_result.txt", "w") as f:
        print(f"CorrelationCalculator (Pearson): {score}")
        f.write(f"CorrelationCalculator (Pearson): {score}\n")

    # Spearman Correlation
    score = metric.evaluate(method=CorrelationMethod.SPEARMAN)
    correlation_pairs = metric.correlation_pairs(method=CorrelationMethod.SPEARMAN)
    orig_corr, syn_corr = correlation_pairs
    orig_corr.to_csv(path / "CorrelationCalculator_Spearman_original_correlation.csv", index=False)
    syn_corr.to_csv(path / "CorrelationCalculator_Spearman_synthetic_correlation.csv", index=False)
    # Create Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(orig_corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Original Data Correlation Heatmap (Spearman)")
    plt.savefig(path / "CorrelationCalculator_Spearman_original_correlation_heatmap.png")
    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.heatmap(syn_corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Synthetic Data Correlation Heatmap (Spearman)")
    plt.savefig(path / "CorrelationCalculator_Spearman_synthetic_correlation_heatmap.png")
    plt.clf()
    with open(path / f"CorrelationCalculator_Spearman_result.txt", "w") as f:
        f.write(f"CorrelationCalculator (Spearman): {score}\n")
        print(f"CorrelationCalculator (Spearman): {score}")

    # Plot synthetic vs original datapoints in 2d space
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    combined_data = pd.concat([full_train, synthetic_data], ignore_index=True)
    pca_result = pca.fit_transform(combined_data.select_dtypes(include=[np.number]).fillna(0))
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:len(full_train), 0], pca_result[:len(full_train), 1], label='Original', alpha=0.5)
    plt.scatter(pca_result[len(full_train):, 0], pca_result[len(full_train):, 1], label='Synthetic', alpha=0.5)
    plt.title("PCA of Original vs Synthetic Data")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(path / "PCA_Original_vs_Synthetic.png")
    plt.clf()

    print(model)