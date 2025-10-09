import logging
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ADULT_CATEGORICAL_COLUMNS = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
]

ADULT_TARGET = "income"

class DatasetLoader:
    def __init__(self, dataset_dir: str, categorical_cols: list[str] = None, target: str = None):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            raise ValueError(f"Dataset directory {self.dataset_dir} does not exist or is not a directory.")

        # train and test files
        self.train_file = self.dataset_dir / 'train.csv'
        self.test_file = self.dataset_dir / 'test.csv'

        self._dataframes = {
            'train': pd.read_csv(self.train_file, header='infer'),
            'test': pd.read_csv(self.test_file, header='infer')
        }

        # Convert int to Int
        for key in self._dataframes:
            int_cols = self._dataframes[key].select_dtypes(include=["int"]).columns
            self._dataframes[key][int_cols] = self._dataframes[key][int_cols].astype("Int64")

        # Convert specified columns to categorical
        if categorical_cols:
            self.categorical_cols = categorical_cols
            for col in categorical_cols:
                if col in self._dataframes['train'].columns:
                    self._dataframes['train'][col] = self._dataframes['train'][col].astype('category')
                if col in self._dataframes['test'].columns:
                    self._dataframes['test'][col] = self._dataframes['test'][col].astype('category')

        self.target = target

    def get_train(self) -> pd.DataFrame:
        return self._dataframes['train']

    def get_test(self) -> pd.DataFrame:
        return self._dataframes['test']

    def split_noniid(self, n: int, seed: int = 42) -> list[pd.DataFrame]:
        df = self.get_train()
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        save_dir = self.dataset_dir / 'splits' / str(n)
        save_dir.mkdir(parents=True, exist_ok=True)

        split_size = len(df) // n
        splits = []
        for i in range(n):
            start = i * split_size
            end = (i + 1) * split_size if i < n - 1 else len(df)
            part_df = df.iloc[start:end].copy()
            part_path = save_dir / f'train_part_{i}.csv'
            part_df.to_csv(part_path, index=False)
            splits.append(df.iloc[start:end].copy())

        return splits

    def split_iid(self, nr_agents: int, seed: int = 42):
        # TODO: Ensure each model gets roughly the same amount of rows
        df = self.get_train()
        np.random.seed(seed)
        splits = [[] for _ in range(nr_agents)]
        for label, group in df.groupby(self.categorical_cols):
            group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
            n = len(group)

            if n < nr_agents:
                agent = np.random.randint(nr_agents)
                splits[agent].append(group)
            else:
                parts = np.array_split(group, nr_agents)
                for i in range(nr_agents):
                    splits[i].append(parts[i])

        splits = [pd.concat(split).reset_index(drop=True) for split in splits]

        # Print nr categories for each split, each group

        return splits


if __name__ == "__main__":

    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income"
    ]
    simple_loader = DatasetLoader(dataset_dir="../../data/adult/csv", categorical_cols=ADULT_CATEGORICAL_COLUMNS, target=ADULT_TARGET)
    df_train = simple_loader.get_train()
    df_test = simple_loader.get_test()
    split = simple_loader.split_iid(nr_agents=3)
    print("\nSimple loader train shape:", df_train.shape)
    print("Simple loader test shape:", df_test.shape)
    print("Simple loader split lengths:", [len(part) for part in split])


