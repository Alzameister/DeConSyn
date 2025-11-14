import argparse
import gc
import re
import sys
from pathlib import Path

from DeConSyn.data.data_loader import DatasetLoader
from DeConSyn.evaluation.evaluator import Evaluator
from DeConSyn.pipelines.config import load_config

def format_run_name(run_dir_name):
    parts = run_dir_name.split('-')
    agents = re.search(r'(\d+)Agents', run_dir_name)
    epochs = re.search(r'(\d+)Epochs', run_dir_name)
    iterations = re.search(r'(\d+)Iterations', run_dir_name)
    # Get topology from 'run-20251103-120213-4Agents-1Epochs-300Iterations-ring-ctgan' --> 'ring', or 'smallworld' etc.
    topology = re.search(r'(ring|full|smallworld)', run_dir_name)
    full = 'Full' if 'full' in run_dir_name.lower() else ''
    model_type = parts[-1].upper()

    formatted = f"{agents.group(1)}A {epochs.group(1)}E {iterations.group(1)}R {topology.group(1).capitalize()} {model_type}".strip()
    return formatted

def eval_agents(config):
    run_dir = Path(config["dir"])
    agent_dirs = list(run_dir.rglob("agent_*"))
    if not agent_dirs:
        print("ERROR: No agents found in run-dir", file=sys.stderr)
        sys.exit(2)

    original_data_path = config['original_data_path']
    baseline_dir = config['baseline_dir']
    categorical_columns = config['categorical_columns']
    target = config['target']
    loader = DatasetLoader(original_data_path, categorical_columns, target)
    original_data = loader.get_train()
    test_data = loader.get_test()

    for agent_dir in agent_dirs:
        agent_nr = agent_dir.name.split('_')[-1]
        #if not agent_nr.startswith('00'):
        #    continue
        run_dir_name = agent_dir.parent.name
        if "5Agents" in run_dir_name or "6Agents" in run_dir_name:
            continue
        model_type = run_dir_name.split('-')[-1]
        print("Evaluating agent:", agent_dir.name, "from run:", run_dir_name)
        synthetic_name = format_run_name(run_dir_name)

        i = config["iterations"]
        while True:
            if model_type == "ctgan":
                #model_name = f"iter-{i:05d}-model.pkl"
                model_name = f"iter-{i:05d}-weights.pt"
            elif model_type == "tabddpm":
                model_name = f"iter-{i:05d}-weights.pt"
            model_path = agent_dir / model_name
            if not model_path.exists():
                break

            print(f"Evaluating model: {model_name}")
            out_dir = agent_dir / f'results-iter-{i:05d}'
            out_dir.mkdir(parents=True, exist_ok=True)

            evaluator = Evaluator(original_data=original_data,
                                  original_data_path=original_data_path,
                                  categorical_columns=categorical_columns,
                                  agent_dir=str(agent_dir),
                                  metrics=config['metrics'],
                                  model_type=model_type,
                                  model_name=model_name,
                                  dataset_name=config['dataset_name'],
                                  synthetic_name=synthetic_name,
                                  keys=config['keys'],
                                  target=target,
                                  seed=config['seed'],
                                  iteration=i,
                                  baseline_dir=baseline_dir,
                                  test_data=test_data
                                  )
            results = evaluator.evaluate()
            print(f"\n========= SUMMARY for {agent_dir.name} ({model_name}) =========\n")
            print(results)
            i += config["iterations"]

            del results
            del evaluator
            if 'torch' in globals():
                import torch
                torch.cuda.empty_cache()
            gc.collect()

def eval_baseline(config):
    original_data_path = config['original_data_path']
    categorical_columns = config['categorical_columns']
    target = config['target']
    loader = DatasetLoader(original_data_path, categorical_columns, target)
    original_data = loader.get_train()
    test_data = loader.get_test()
    baseline_dir = config['baseline_dir']
    metrics = config['metrics']
    if 'Consensus' in metrics:
        # Remove consensus from metrics
        metrics.remove('Consensus')


    evaluator = Evaluator(
        original_data=original_data,
        original_data_path=original_data_path,
        categorical_columns=categorical_columns,
        agent_dir=baseline_dir,
        metrics=config['metrics'],
        model_type=config['model_type'],
        model_name=config['baseline_model_name'],
        dataset_name=config['dataset_name'],
        synthetic_name=config['dataset_name'],
        keys=config['keys'],
        target=target,
        seed=config['seed'],
        baseline=True,
        test_data=test_data
    )
    results = evaluator.evaluate()
    print(f"\n========= SUMMARY for Baseline ({config['baseline_model_name']}) =========\n")
    print(results)

def eval_iterations(config):
    iterations = config['iterations']
    steps = 100
    for i in range(steps, iterations + 1, steps):
        print(f"\nEvaluating iteration: {i}/{iterations}\n")
        config['iterations'] = i
        eval_agents(config)



def main():
    parser = argparse.ArgumentParser(description="Evaluate experiments in a given directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation configuration file.")
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline.")
    parser.add_argument("--iterations", action="store_true", help="Evaluate across iterations.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.baseline:
        eval_baseline(config)
    elif args.iterations:
        eval_iterations(config)
    else:
        eval_agents(config)

if __name__ == "__main__":
    main()