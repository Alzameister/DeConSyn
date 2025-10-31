import json
from pathlib import Path

import numpy as np
import pandas as pd
from DeConSyn.data.data_loader import ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET


class DataTransformer:
    def __init__(self, data_dir: str, target: str, cat_cols: list):
        self.data_dir = Path(data_dir).expanduser().resolve()
        print("Data directory:", self.data_dir)
        self.columns = self._get_columns()
        self.target = target
        # Remove target from cat_cols if present
        self.cat_cols = cat_cols if target not in cat_cols else [col for col in cat_cols if col != target]
        self.num_cols = [col for col in self.columns if col not in self.cat_cols + [self.target]]
        self.train_file = self._find_file(['train.csv', '*.train'])
        self.test_file = self._find_file(['test.csv', '*.test'])
        self.save_csv(self.data_dir / 'csv')
        self.save_npy(self.data_dir / 'npy')
        self.save_info(self.data_dir / 'npy')

    def _find_file(self, patterns):
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                return files[0]
        return None

    def _get_columns(self):
        names_file = next(self.data_dir.glob('*.names'), None)
        if names_file:
            print(
                f"Using column names from {names_file}"
            )
            columns = []
            with open(names_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('|'):
                        continue
                    if ':' in line:
                        name, _ = line.split(':', 1)
                        columns.append(name.strip())
            return columns
        return None

    def _clean(self, file_path):
        df = pd.read_csv(
            file_path,
            header=None if self.columns else 'infer',
            names=self.columns,
            index_col=False
        )
        int_cols = df.select_dtypes(include=["int"]).columns
        df[int_cols] = df[int_cols].astype("Int64")
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lstrip()
        df.replace('?', 'MISSING', inplace=True)
        return self._align_data(df)

    def get_train(self):
        if self.train_file is None:
            raise FileNotFoundError("No train file found.")
        return self._clean(self.train_file)

    def get_test(self):
        if self.test_file is None:
            raise FileNotFoundError("No test file found.")
        return self._clean(self.test_file)

    def _align_data(self, df):
        ordered_columns = self.num_cols + self.cat_cols + [self.target]
        return df[ordered_columns]


    def save_csv(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df = self.get_train()
        train_df.to_csv(output_dir / 'train.csv', index=False)
        try:
            test_df = self.get_test()
            test_df.to_csv(output_dir / 'test.csv', index=False)
        except FileNotFoundError:
            pass

    def save_npy(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df = self.get_train()
        try:
            test_df = self.get_test()
        except FileNotFoundError:
            test_df = None

        num_cols = [col for col in train_df.columns if col not in self.cat_cols + [self.target]]


        np.save(output_dir / "X_num_train.npy", train_df[num_cols].to_numpy(dtype=np.float32))
        np.save(output_dir / "X_cat_train.npy",
                train_df[self.cat_cols].to_numpy())
        np.save(output_dir / "y_train.npy", train_df[self.target].to_numpy())

        if test_df is not None:
            np.save(output_dir / "X_num_test.npy", test_df[num_cols].to_numpy(dtype=np.float32))
            np.save(output_dir / "X_cat_test.npy",
                    test_df[self.cat_cols].to_numpy())
            np.save(output_dir / "y_test.npy", test_df[self.target].to_numpy())

    @staticmethod
    def save_split_info(
            df: pd.DataFrame,
            output_dir: Path,
            cat_cols: list,
            target: str
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        y = df[target]
        if y.dtype.name == 'category' or y.dtype == object:
            n_classes = int(y.nunique())
            if n_classes == 2:
                task_type = 'binclass'
            else:
                task_type = 'classification'
        else:
            task_type = "regression"
            n_classes = None

        info = {
            "task_type": task_type,
            "n_classes": n_classes,
            "num_features": len([col for col in df.columns if col not in cat_cols + [target]]),
            "cat_features": cat_cols,
            "target": target
        }
        with open(output_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    @staticmethod
    def save_split_npy(
            df: pd.DataFrame,
            output_dir: Path,
            i: int,
            cat_cols: list,
            target: str
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        num_cols = [col for col in df.columns if col not in cat_cols + [target]]

        out_dir = output_dir / f'split_{i}'
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'X_num_train.npy', df[num_cols].to_numpy(dtype=np.float32))
        np.save(out_dir / "X_cat_train.npy", df[cat_cols].to_numpy())
        np.save(out_dir / "y_train.npy", df[target].to_numpy())

        DataTransformer.save_split_info(
            df,
            out_dir,
            cat_cols,
            target
        )

    @staticmethod
    def save_test_split_npy(
            df: pd.DataFrame,
            output_dir: Path,
            i: int,
            cat_cols: list,
            target: str
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        num_cols = [col for col in df.columns if col not in cat_cols + [target]]

        out_dir = output_dir / f'split_{i}'
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'X_num_test.npy', df[num_cols].to_numpy(dtype=np.float32))
        np.save(out_dir / "X_cat_test.npy", df[cat_cols].to_numpy())
        np.save(out_dir / "y_test.npy", df[target].to_numpy())

        DataTransformer.save_split_info(
            df,
            out_dir,
            cat_cols,
            target
        )


    def save_info(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df = self.get_train()
        y = train_df[self.target]
        if y.dtype.name == 'category' or y.dtype == object:
            n_classes = int(y.nunique())
            if n_classes == 2:
                task_type = 'binclass'
            else:
                task_type = 'classification'
        else:
            task_type = "regression"
            n_classes = None

        info = {
            "task_type": task_type,
            "n_classes": n_classes,
            "num_features": len([col for col in train_df.columns if col not in self.cat_cols + [self.target]]),
            "cat_features": self.cat_cols,
            "target": self.target
        }
        with open(output_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    transformer = DataTransformer("../../data/adult", ADULT_TARGET, ADULT_CATEGORICAL_COLUMNS)
    df_train = transformer.get_train()
    df_test = transformer.get_test()
    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape if df_test is not None else "No test file found")
