from DeConSyn.data.data_loader import ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET, DatasetLoader
from DeConSyn.io.io import get_repo_root

repo_root = get_repo_root()
split_dir = repo_root / 'data' / 'adult' / 'csv' / 'splits'
data_root = repo_root / 'data' / 'adult'

loader = DatasetLoader(str(data_root), ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET)
train = loader.get_train()
splits = loader.split_iid(4, 42)
for i, split in enumerate(splits):
    # Save each split
    split_path = split_dir / '4' / f'train_part_{i}.csv'
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False)

# Plot distributions of each split
import math
import matplotlib.pyplot as plt
import seaborn as sns

for i, split in enumerate(splits):
    split_path = split_dir / '4'
    split_path.mkdir(parents=True, exist_ok=True)

    cols = list(split.columns)
    ncols = min(4, len(cols)) or 1
    nrows = math.ceil(len(cols) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 4, nrows * 3),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, col in zip(axes_flat, cols):
        if col in ADULT_CATEGORICAL_COLUMNS + [ADULT_TARGET]:
            sns.countplot(data=split, x=col, ax=ax)
        else:
            sns.histplot(data=split, x=col, kde=True, ax=ax)
        ax.set_title(f'{col} Distribution')

    for ax in axes_flat[len(cols):]:
        ax.axis('off')

    fig.suptitle(f'4 Splits - Split {i} - All Columns', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(split_path / f'split_{i}.png')
    plt.close(fig)

