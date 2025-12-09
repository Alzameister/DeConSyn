# DeConSyn: Fake it till you make it. Design and Implementation of a Decentralized Synthetic Data Generation Framework

Protecting the privacy of data by allowing decentralized training of synthetic data generators, enabling collaborative data synthesis without sharing raw data.

## Setup

1. Install and configure an XMPP server (e.g., [openfire](https://www.igniterealtime.org/projects/openfire/))
2. Install [poetry](https://python-poetry.org/)
3. Run the following command:

```bash
poetry install
```

## Running experiments

### Datasets

The datasets can be found under `/data`. Adult, cardio, and churn datasets were used in this thesis.

### Used models

- [CTGAN](https://github.com/sdv-dev/CTGAN): A GAN-based model for generating tabular data.
- [TabDDPM](https://github.com/yandex-research/tab-ddpm): A diffusion-based model for generating tabular data.

### File structure

- `data/`: Contains datasets used for experiments.
- `DeConSyn/`: Main package containing the implementation of the decentralized synthetic data generation framework.
- `DeConSyn/pipelines/`: Contains the necessary pipelines to run experiments and evaluate the trained models.
- `exp/`: Contains configuration files for different datasets.

The following scripts are available to run and evaluate the experiments:
- `DeConSyn/pipelines/run_exp.py`: Script to run the decentralized synthetic data generation experiments. It takes a path to a configuration file as an argument.
- `DeConSyn/pipelines/evaluate_exp.py`: Script to evaluate the trained models. It takes a path to a configuration file as an argument.

### Configurations

Configuration files for different datasets are located in the `exp/` directory. The configuration file for training a model specifies the parameters for the experiments, including dataset paths and details of model hyperparameters for DeConSyn and the underlying synthetic data generator. An example can be found in `exp/adult/ctgan_config.yaml`.  The configuration file for evaluation specifies the parameters for evaluating the trained models, including paths to the trained models and evaluation metrics. An example can be found in `exp/adult/ctgan_eval_config.yaml`. The schema of the configuration files is described in detail in [CONFIG_DESCRIPTION.md](CONFIG_DESCRIPTION.md).

### Examples

To run an experiment, call the following command:

```bash
poetry run python DeConSyn/pipelines/run_exp.py --config_path exp/adult/ctgan_config.yaml
```

To evaluate a trained model, call the following command:

```bash
poetry run python DeConSyn/pipelines/evaluate_exp.py --config_path exp/adult/ctgan_eval_config.yaml
```

## Results

The raw and intermediate results of the experiments can be found [here](https://1drv.ms/f/c/6b24e7b6af3b34c6/EuH8Ltl1aZZJt2xZvJKYkk8BxkfKrD41zyEgERNcxPouRw?e=76vsEL).

Each dataset has its own folder containing the results and logs for the experiments conducted on that dataset.

Each dataset contains a folder `aggregate_results`, which contains the results averaged over all agents of each individual run (`{model_type}-{iter}.csv`), aggregate metrics over all reruns of once configuration (`{model_type}-{iter}-group-stats.csv`), aggregate across all configurations (`{model_type}-{iter}-overall-metric-stats.csv`), aggregates across median configurations (`{model_type}-{iter}-agg-across-groups.csv`), and 95% confidence intervals for each configuration (`{model_type}-{iter}-ci.csv`).

Each dataset folder also contains subfolders for logs and all individual runs of each configuration and for each model type, as well as the centralized baselines of each model type, and a subfolder for FedTabDiff baseline experiments.

Inside the run folders, each individual run has its own folder containing the weights files, generated synthetic data, and results of the evaluation metrics for each agent participating in that run, which also includes some plots like PCA and distribution plots. Please note that these folders can be very (!) large, especially for diffusion-based models, as the weights have been saved periodically for each individual agent during training to track the convergence of the different agents. The run folders of each configuration have been compressed to save space, and it is suggested to only download and decompress the configurations that are of interest.

The aggregated results over agent counts and topology can be found under `topology_agents_results` in the parent folder.
