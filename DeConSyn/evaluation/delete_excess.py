import re
import glob
import os
import shutil
from pathlib import Path

ITER_RE = re.compile(r"iter-(\d+)-weights\.pt$")
RESULTS_RE = re.compile(r"^results.*")


def cleanup_checkpoints(dir_path):
    files = glob.glob(str(Path(dir_path) / "**" / "iter-*-weights.pt"), recursive=True)
    for f in files:
        m = ITER_RE.search(os.path.basename(f))
        if m:
            iter_num = int(m.group(1))
            if iter_num % 10 != 0 and iter_num != 1:
                os.remove(f)
                print(f"Deleted: {f}")

def cleanup_results(dir_path):
    for root, dirs, _ in os.walk(dir_path):
        for d in dirs:
            if RESULTS_RE.match(d):
                dir_to_remove = os.path.join(root, d)
                shutil.rmtree(dir_to_remove)
                print(f"Deleted directory: {dir_to_remove}")

def cleanup_distributions(dir_path):
    for root, dirs, _ in os.walk(dir_path):
        for d in dirs:
            if d == "Distributions":
                dir_to_remove = os.path.join(root, d)
                shutil.rmtree(dir_to_remove)
                print(f"Deleted directory: {dir_to_remove}")

# Usage: replace 'your_dir_path' with your directory
#cleanup_results('C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/tabddpm/')
cleanup_distributions('C:/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs/ctgan/')