import argparse
import re
import sys
from pathlib import Path

from DeFeSyn.evaluation.evaluator import Evaluator
from DeFeSyn.models.tab_ddpm.lib.util import load_config


def _csv_list(s):
    return [x.strip() for x in s.split(",")] if s else []

def eval_agents(args):
    keys = ["age", "education", "marital-status", "occupation"]
    target = 'income'
    run_dir = Path(args.run_dir)

    # Find all agent directories (also inside nested folders)
    agent_dirs = list(run_dir.rglob("agent_*"))

    if not agent_dirs:
        print("ERROR: No agents found in run-dir", file=sys.stderr)
        sys.exit(2)

    for agent_dir in agent_dirs:
        model_path = agent_dir / args.model_name
        if not model_path.exists():
            print(f"ERROR: Model {model_path} not found in agent dir {agent_dir}.", file=sys.stderr)
            continue

        run_dir_name = agent_dir.parent.name
        print("Evaluating agent:", agent_dir.name, "from run:", run_dir_name)
        model_type = run_dir_name.split('-')[-1]

        out_dir = agent_dir / 'results'
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic_name = format_run_name(run_dir_name)
        metrics = ['DCR', "Disclosure", 'NNDR', 'BasicStats', 'PCA']

        evaluator = Evaluator(
            data_dir=args.data_dir,
            categorical_cols=args.categorical_cols,
            model_path=model_path,
            output_dir=str(out_dir),
            keys=keys,
            target=target,
            model_type=model_type,
            synthetic_name=synthetic_name,
            metrics=metrics,
            run_dir=str(agent_dir)
        )
        evaluator.seed = 42
        results = evaluator.evaluate()
        print(f"\n========= SUMMARY for {agent_dir.name} =========\n")
        print(results)

def eval_agents_it(args, iterations):
    keys = ["age", "education", "marital-status", "occupation"]
    target = 'income'
    run_dir = Path(args.run_dir)

    # Find all agent directories (also inside nested folders)
    agent_dirs = list(run_dir.rglob("agent_*"))

    if not agent_dirs:
        print("ERROR: No agents found in run-dir", file=sys.stderr)
        sys.exit(2)

    for agent_dir in agent_dirs:
        model_path = agent_dir / args.model_name
        if not model_path.exists():
            print(f"ERROR: Model {model_path} not found in agent dir {agent_dir}.", file=sys.stderr)
            continue

        run_dir_name = agent_dir.parent.name
        print("Evaluating agent:", agent_dir.name, "from run:", run_dir_name)
        model_type = run_dir_name.split('-')[-1]


        synthetic_name = format_run_name(run_dir_name)
        # metrics = ['DCR', "Disclosure", 'NNDR', 'BasicStats', 'PCA', 'Consensus']
        metrics = ['Consensus']
        #metrics = ['AdversarialAccuracy', "JS", "KS", "CorrelationPearson", "CorrelationSpearman"]

        i = iterations
        while True:
            if model_type == "ctgan":
                model_name = f"iter-{i:05d}-model.pkl"
            elif model_type == "tabddpm":
                model_name = f"iter-{i:05d}-weights.pt"
            model_path = agent_dir / model_name
            if not model_path.exists():
                break
            print(f"Evaluating model: {model_name}")
            out_dir = agent_dir / f'results-iter-{i:05d}'
            out_dir.mkdir(parents=True, exist_ok=True)
            evaluator = Evaluator(
                data_dir=args.data_dir,
                categorical_cols=args.categorical_cols,
                model_path=str(model_path),
                output_dir=str(out_dir),
                keys=keys,
                target=target,
                model_type=model_type,
                synthetic_name=synthetic_name,
                metrics=metrics,
                run_dir=str(agent_dir)
                            )
            evaluator.seed = 42
            results = evaluator.evaluate()
            print(f"\n========= SUMMARY for {agent_dir.name} ({model_name}) =========\n")
            print(results)
            i += iterations


def main():
    parser = argparse.ArgumentParser(description="Evaluate all agents in a run directory (no manifest required).")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the original datasets.")
    parser.add_argument("--categorical-cols", type=_csv_list, default=[], help="Comma-separated list of categorical column names.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model file to load from each agent directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing agent subdirectories (agent_0, agent_1, ...).")
    parser.add_argument("--iterations", nargs="?", type=int, const=None, help="If set, use iterative agent evaluation with the given integer.")

    args = parser.parse_args()
    if args.iterations is None:
        eval_agents(args)
    else:
        eval_agents_it(args, args.iterations)

def format_run_name(run_dir_name):
    parts = run_dir_name.split('-')
    agents = re.search(r'(\d+)Agents', run_dir_name)
    epochs = re.search(r'(\d+)Epochs', run_dir_name)
    iterations = re.search(r'(\d+)Iterations', run_dir_name)
    full = 'Full' if 'full' in run_dir_name.lower() else ''
    model_type = parts[-1].upper()

    formatted = f"{agents.group(1)}A {epochs.group(1)}E {iterations.group(1)}R {full} {model_type}".strip()
    return formatted

if __name__ == "__main__":
    main()
