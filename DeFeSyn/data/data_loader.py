from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder

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
        # TODO: Also get numpy versions, fit ohe / ordinal encoders
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
        self.categorical_cols = categorical_cols
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in self._dataframes['train'].columns:
                    self._dataframes['train'][col] = self._dataframes['train'][col].astype('category')
                if col in self._dataframes['test'].columns:
                    self._dataframes['test'][col] = self._dataframes['test'][col].astype('category')

        self.target = target
        if self.target:
            if self.target in self._dataframes['train'].columns:
                self._dataframes['train'][self.target] = self._dataframes['train'][self.target].astype('category')
            if self.target in self._dataframes['test'].columns:
                self._dataframes['test'][self.target] = self._dataframes['test'][self.target].astype('category')

        self.numerical_cols = [col for col in self._dataframes['train'].columns
                               if col not in self.categorical_cols and col != self.target]


    def get_train(self) -> pd.DataFrame:
        return self._dataframes['train']

    def get_test(self) -> pd.DataFrame:
        return self._dataframes['test']

    def split_iid(self, nr_agents: int, seed: int = 42):
        df = self.get_train()
        rng = np.random.default_rng(seed)
        allocations = [[] for _ in range(nr_agents)]
        assigned_indices = set()

        # Ensure each agent gets at least one row per value in each categorical column
        for col in self.categorical_cols:
            for val, group in df.groupby(col, observed=True, sort=False):
                idx = group.index.difference(assigned_indices).to_numpy()
                rng.shuffle(idx)
                for i in range(min(nr_agents, len(idx))):
                    allocations[i].append(idx[i])
                    assigned_indices.add(idx[i])

        remaining_indices = df.index.difference(assigned_indices).to_numpy()
        rng.shuffle(remaining_indices)
        split_sizes = [len(remaining_indices) // nr_agents] * nr_agents
        for i in range(len(remaining_indices) % nr_agents):
            split_sizes[i] += 1

        start = 0
        for i, size in enumerate(split_sizes):
            end = start + size
            allocations[i].extend(remaining_indices[start:end])
            start = end

        splits = []
        for agent_rows in allocations:
            agent_df = df.loc[agent_rows].sample(frac=1.0, random_state=seed)
            splits.append(agent_df.reset_index(drop=True))

        return splits

    def _fit_ohe(self):
        df = self.get_train()
        df = df[self.categorical_cols + [self.target]]
        # Encode cat columns
        self.ohe = OneHotEncoder().fit(df)

        return self.ohe




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
    ohe = simple_loader._fit_ohe()
    split = simple_loader.split_iid(nr_agents=3)
    print("\nSimple loader train shape:", df_train.shape)
    print("Simple loader test shape:", df_test.shape)
    print("Simple loader split lengths:", [len(part) for part in split])


