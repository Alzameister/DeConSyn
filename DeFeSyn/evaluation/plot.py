import ast
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_runs_path():
    # Get repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runs_path = os.path.join(repo_root, 'runs')
    if not os.path.exists(runs_path):
        return None
    return runs_path

def get_runs_dir():
    path = get_runs_path()
    if path is None:
        return None
    return os.listdir(path)

def get_runs_results_df(it: int = 500):
    # Get results of each agent in each run
    runs_path = get_runs_path()
    runs = get_runs_dir()
    if runs is None:
        return None

    results = []
    for run in runs:
        if not os.path.isdir(os.path.join(runs_path, run)):
            continue
        run_path = os.path.join(get_runs_path(), run)
        # Get all subruns
        run_attempts = [d for d in os.listdir(run_path)]

        for run_attempt in run_attempts:
            run_attempt_path = os.path.join(run_path, run_attempt)
            if not os.path.isdir(run_attempt_path):
                continue

            # Get all dirs that start with "agent_"
            agents = [d for d in os.listdir(run_attempt_path) if d.startswith('agent_')]
            for agent in agents:
                agent_path = os.path.join(run_attempt_path, agent)
                results_file = f"results_iter_{it:04d}/results.csv"
                result_file = os.path.join(agent_path, results_file)
                if os.path.exists(result_file):
                    df = pd.read_csv(result_file)
                    df['run'] = run_attempt
                    df['agent'] = agent
                    clean_disclosure(df)
                    clean_basicstats(df)
                    results.append(df)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()

def clean_disclosure(df: pd.DataFrame) -> pd.DataFrame:
    if 'Disclosure' in df.columns:
        disclosure = df['Disclosure'].str.replace('[\[\]]', '', regex=True)
        if disclosure.isnull().all():
            df.drop(columns=['Disclosure'], inplace=True)
            return df
        disclosure_dict = df['Disclosure'].apply(safe_eval)
        keys = disclosure_dict.apply(lambda d: list(d.keys()))
        values = disclosure_dict.apply(lambda d: list(d.values()))
        # Set keys as columns with values
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                key = keys[i][j]
                value = values[i][j]
                df.at[i, key] = value
        df.drop(columns=['Disclosure'], inplace=True)
    return df

def clean_basicstats(df: pd.DataFrame) -> pd.DataFrame:
    if 'BasicStats' in df.columns:
        basicstats = df['BasicStats'].str.replace('[\[\]]', '', regex=True)
        if basicstats.isnull().all():
            df.drop(columns=['BasicStats'], inplace=True)
            return df
        basicstats_dict = df['BasicStats'].apply(safe_eval)
        keys = basicstats_dict.apply(lambda d: list(d.keys()))
        values = basicstats_dict.apply(lambda d: list(d.values()))
        # Set keys as columns with values
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                key = keys[i][j].capitalize()
                value = values[i][j]
                df.at[i, key] = value
        df.drop(columns=['BasicStats'], inplace=True)
    return df

def safe_eval(x):
    if pd.isna(x):
        return {}
    return ast.literal_eval(x)


def get_avg_agents_df(it: int = 500):
    df = get_runs_results_df(it)
    if df is None or df.empty:
        return None
    numeric_cols = df.select_dtypes(include='number').columns
    avg_df = df.groupby(['run'])[numeric_cols].mean().reset_index()
    return avg_df

def get_avg_runs_df(it: int = 500):
    df = get_avg_agents_df(it)
    baseline_df = get_baseline_df()
    baseline_df['run'] = 'baseline_ctgan'
    # Merge df and baseline_df on 'run' column
    if baseline_df is not None and not baseline_df.empty:
        df = pd.concat([baseline_df, df], ignore_index=True)
    if df is None or df.empty:
        return None
    numeric_cols = df.select_dtypes(include='number').columns
    # Avg over 'run' column, remove until first '-'
    df['run_base'] = df['run'].apply(lambda x: x.partition('-')[2] if '-' in x else x)
    avg_df = df.groupby(['run_base'])[numeric_cols].mean().reset_index()
    avg_df = avg_df.rename(columns={'run_base': 'run'})
    return avg_df

def get_baseline_df():
    runs_path = get_runs_path()
    baseline_path = os.path.join(runs_path, 'baseline_ctgan', 'results', 'results.csv')
    if os.path.exists(baseline_path):
        return pd.read_csv(baseline_path)
    return None

def get_run_parameters(run_name: str) -> dict:
    # Example: '01-7A-1E-500R-Full'
    parts = run_name.split('-')
    if len(parts) >= 4:
        agents = ''.join(filter(str.isdigit, parts[0]))
        epochs = ''.join(filter(str.isdigit, parts[1]))
        rounds = ''.join(filter(str.isdigit, parts[2]))
        return (agents, epochs, rounds)
    return ('baseline', '', '')

def get_run_topology(run_name: str) -> str:
    parts = run_name.split('-')
    if len(parts) >= 4:
        return parts[3].capitalize()
    return 'Baseline'

def plot_avg(it: int = 500):
    df = get_avg_runs_df(it)
    runs_path = get_runs_path()
    plots_dir = os.path.join(runs_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)


    if df is None or df.empty:
        print("No data to plot.")
        return

    # Plot per metric
    df['params'] = df['run'].apply(get_run_parameters)
    numeric_cols = df.select_dtypes(include='number').columns
    # Rename columns for better readability with getting run topolgy
    df['Topology'] = df['run'].apply(get_run_topology)
    baseline_row = df[df['run'] == 'baseline_ctgan']

    for params, group in df.groupby('params'):
        if not baseline_row.empty and 'baseline_ctgan' not in group['run'].values:
            group = pd.concat([baseline_row, group], ignore_index=True)
        param_str = f"Agents: {params[0]}, Epochs: {params[1]}, Rounds: {params[2]}"
        for col in numeric_cols:
            plt.figure()
            plt.plot(group['Topology'], group[col], marker='o')
            plt.title(f'Average {col} per Run\n{param_str}')
            plt.xlabel('Run')
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            fname = f"avg_{col}_A{params[0]}_E{params[1]}_R{params[2]}.png"
            plt.savefig(os.path.join(plots_dir, fname))
            plt.close()

