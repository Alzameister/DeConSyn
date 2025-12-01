import pandas as pd

from DeConSyn.evaluation.utility import LogisticRegressionEvaluator, CatBoostEvaluator
from DeConSyn.io.io import get_repo_root
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
    CorrelationCalculator, CorrelationMethod
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.js_similarity import \
    JSCalculator
from FEST.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import \
    KSCalculator

repo_root = get_repo_root()
fedtabdiff_eval_dir = repo_root / 'exp' / 'adult' / 'runs' / 'FedTabDiff' / '10A'
original_dir = repo_root / 'data' / 'adult' / 'csv'
dataset_name = 'adult'
synthetic_name = 'FedTabDiff'
keys = ["age", "education", "marital-status", "occupation"]
target = 'income'

synthetic = pd.read_csv(fedtabdiff_eval_dir / 'synthetic.csv')

original = pd.read_csv(original_dir / 'train.csv')
for col in original.select_dtypes(include=['int64']).columns:
    original[col] = original[col].astype('float64')

results_file = fedtabdiff_eval_dir / 'results.csv'
results = pd.DataFrame()

dcr_calculator = DCRCalculator(original=original, synthetic=synthetic,
                                                   original_name=dataset_name, synthetic_name=synthetic_name)
dcr_value = dcr_calculator.evaluate()
results.at[synthetic_name, 'DCR'] = dcr_value
nndr_calculator = NNDRCalculator(original=original, synthetic=synthetic,
                                                   original_name=dataset_name, synthetic_name=synthetic_name)
nndr_value = nndr_calculator.evaluate()
results.at[synthetic_name, 'NNDR'] = nndr_value
disco_calculator = DisclosureCalculator(original=original, synthetic=synthetic,
                                                   original_name=dataset_name, synthetic_name=synthetic_name,
                                        keys=keys, target=target)
repu_value, disco_value = disco_calculator.evaluate()
results.at[synthetic_name, 'Disclosure'] = disco_value
results.at[synthetic_name, 'RepU'] = repu_value

aa_calculator = AdversarialAccuracyCalculator(original=original, synthetic=synthetic,
                                              original_name=dataset_name, synthetic_name=synthetic_name)
aa_value = aa_calculator.evaluate()
results.at[synthetic_name, 'AdversarialAccuracy'] = aa_value

basicstats_calculator = BasicStatsCalculator(original=original, synthetic=synthetic,
                                                   original_name=dataset_name, synthetic_name=synthetic_name)
stats_results = basicstats_calculator.evaluate()
for stat_name, stat_value in stats_results.items():
    results.at[synthetic_name, stat_name.capitalize()] = stat_value

js_calculator = JSCalculator(original=original, synthetic=synthetic,
                                                   original_name=dataset_name, synthetic_name=synthetic_name)
js_results = js_calculator.evaluate()
results.at[synthetic_name, 'JS'] = js_results

ks_calculator = KSCalculator(original=original, synthetic=synthetic,
                                                   original_name=dataset_name, synthetic_name=synthetic_name)
ks_results = ks_calculator.evaluate()
results.at[synthetic_name, 'KS'] = ks_results

correlation_calculator = CorrelationCalculator(original=original, synthetic=synthetic,
                                               original_name=dataset_name, synthetic_name=synthetic_name)
corr_results = correlation_calculator.evaluate(method=CorrelationMethod.PEARSON)
results.at[synthetic_name, 'CorrelationPearson'] = corr_results
corr_results = correlation_calculator.evaluate(method=CorrelationMethod.SPEARMAN)
results.at[synthetic_name, 'CorrelationSpearman'] = corr_results

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    'income'
]

if target in categorical_columns:
    categorical_columns.remove(target)

test = pd.read_csv(original_dir / 'test.csv')
synth_log_reg_evaluator = LogisticRegressionEvaluator(synthetic, test, target, categorical_columns, seed=42)
synt_accuracy, synth_f1 = synth_log_reg_evaluator.evaluate()
results.at[synthetic_name, 'LogReg_Accuracy'] = synt_accuracy
results.at[synthetic_name, 'LogReg_F1'] = synth_f1

cb_evaluator = CatBoostEvaluator(synthetic, test, target, categorical_columns, seed=42)
cb_synt_accuracy, cb_synth_f1 = cb_evaluator.evaluate()
results.at[synthetic_name, 'CatBoost_Accuracy'] = cb_synt_accuracy
results.at[synthetic_name, 'CatBoost_F1'] = cb_synth_f1

results.to_csv(results_file)