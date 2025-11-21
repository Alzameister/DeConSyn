# Description of .toml config for DeConSyn

## Training Configuration

This is a description of the configuration file used for training experiments in DeConSyn. The configuration is separated into three sections: Main part, deconsyn_params, and if needed, TabDDPM hyperparameters.

Main part:
- `seed`: An integer value used to set the random seed for reproducibility.

DeConSyn parameters (`deconsyn_params`):
- `dataset_name`: A string indicating the name of the dataset being used (e.g., "adult", "cardio", "churn").
- `data_root`: A string specifying the path to the root directory where the dataset is located. uses relative paths from the `DeConSyn/exp/` directory.'
- `categorical_columns`: A list of strings representing the names of categorical columns in the dataset.
- `target`: A string representing the target column name in the dataset.
- `n`: An integer specifying the number of agents.
- `epochs`: An integer specifying the number of local training epochs per FSM iteration.
- `iterations`: An integer specifying the number of FSM rounds.
- `topology`: A string indicating the network topology used for agent communication (e.g., "full", "ring").
- `k`: If the small-world topology is used, `k` is an integer specifying the number of neighbors each agent is initially connected to.
- `p`: If the small-world topology is used, `p` is a float specifying the probability of rewiring each edge.
- `alpha`: A float specifying the scaling of parameter updates within asynchronous consensus learning.
- `gen_model_type`: A string indicating the type of synthetic data generator used (e.g., "ctgan", "tabddpm").
- `log_level`: A string specifying the logging level (e.g., "INFO", "DEBUG").

### Optional TabDDPM

If `gen_model_type` is set to "tabddpm", the following additional hyperparameters have to be specified:

Main part:
- `parent_dir`: A string specifying the path to the parent directory where the experiment is located. Uses the relative path from the repo root.
- `real_data_path`: A string specifying the path to the real data file used for training. Uses the relative path from the repo root.
- `num_numerical_features`: An integer value specifying the number of numerical columns in the dataset.
- `model_type`: A string indicating the type of model TabDDPM uses for the reverse process.
- `device`: A string specifying the device used for training (e.g., "cpu", "cuda:0").

Model_params:
- `num_classes`: An integer specifying the number of classes for the target.
- `is_y_cond`: A boolean indicating whether the model is conditioned on the target variable.
- `d_layers`: A list of integers specifying the dimensions of each layer in the model.
- `dropout`: A float specifying the dropout rate used in the model.
- `num_timesteps`: The number of diffusion timesteps.
- `steps`: An integer specifying the number of total training steps.
- `lr`: A float specifying the learning rate used for training.
- `weight_decay`: A float specifying the weight decay used for training.
- `seed`: The seed used for reproducability inside TabDDPM
- `normalization`: A string specifying the type of normalization used in the model (e.g., "quantile").
- `num_nan_policy`: A string indicating how to handle NaN values in the dataset (e.g., "__none__").
- `cat_nan_policy`: A string indicating how to handle NaN values in categorical columns (e.g., "__none__").
- `cat_min_frequency`: A string specifying the minimum frequency for categorical columns (e.g., "__none__").
- `cat_encoding`: A string specifying how categorical columns are encoded (e.g., "__none__").
- `y_policy`: A string indicating how to handle the target variable (e.g., "default").

While evaluation parameters are needed for the TabDDPM package to work, they are not actively used during training. An example configuration file can be found in `exp/adult/tabddpm_config.yaml`.

## Evaluation Configuration

This is a description of the configuration file used for evaluating trained models in DeConSyn. The configuration contains the following parameters:
- `original_data_path`: A string specifying the path to the original data file used for evaluation.
- `categorical_columns`: A list of strings representing the names of categorical columns in the dataset.
- `baseline_dir`: A string specifying the path to the directory containing baseline results for comparison.
- `baseline_model_name`: A string indicating the file name of the baseline model used for comparison (e.g., "ctgan.pt", "tabddpm.pt").
- `dir`: A string specifying the path to the directory containing the trained models to be evaluated. All models in this directory will be evaluated.
- `metrics`: A list of strings representing the names of evaluation metrics to be computed (e.g., "DCR", "NNDR", "JS").
- `model_type`: A string indicating the type of synthetic data generator used (e.g., "ctgan", "tabddpm").
- `dataset_name`: A string indicating the name of the dataset being used (e.g., "adult", "cardio", "churn").
- `keys`: A list of string representing the keys used for disclosure evaluation.
- `target`: A string representing the target column name in the dataset.
- `seed`: An integer value used to set the random seed for reproducibility.
- `iterations`: An integer specifying the FSM round you want to evaluate.

An example configuration file can be found in `exp/adult/ctgan_eval_config.yaml`.