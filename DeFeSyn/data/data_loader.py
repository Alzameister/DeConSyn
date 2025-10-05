import logging
from pathlib import Path
from pprint import pprint

import pandas as pd
import yaml


class DatasetLoader:
    def __init__(self, dataset_dir: str, categorical_cols: list[str] = None):
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
            for col in categorical_cols:
                if col in self._dataframes['train'].columns:
                    self._dataframes['train'][col] = self._dataframes['train'][col].astype('category')
                if col in self._dataframes['test'].columns:
                    self._dataframes['test'][col] = self._dataframes['test'][col].astype('category')

    def get_train(self) -> pd.DataFrame:
        return self._dataframes['train']

    def get_test(self) -> pd.DataFrame:
        return self._dataframes['test']

    def split(self, n: int, seed: int = 42) -> list[pd.DataFrame]:
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


if __name__ == "__main__":
    dataset_metadata_file = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult/manifest.yaml"
    loader = DatasetLoader(manifest_path=dataset_metadata_file)
    print("Available resources:")
    pprint(loader.resource_names())
    print("\nLoading 'adult' resource:")
    df_adult = loader.get('adult-train')
    print(df_adult.head())
    print("\nConcatenating all resources:")
    df_all = loader.concat()
    print(df_all.head())
    print("\nLength of concatenated DataFrame:", len(df_all))
    print("\n Length of 'adult' DataFrame:", len(df_adult))

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
    simple_loader = SimpleDatasetLoader(dataset_dir="../../data/adult/csv", categorical_cols=categorical_columns)
    df_train = simple_loader.get_train()
    df_test = simple_loader.get_test()
    split = simple_loader.split(n=3)
    print("\nSimple loader train shape:", df_train.shape)
    print("Simple loader test shape:", df_test.shape)
    print("Simple loader split lengths:", [len(part) for part in split])


