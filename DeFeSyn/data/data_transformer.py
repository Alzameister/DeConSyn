from pathlib import Path
import pandas as pd

class DataTransformer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.columns = self._get_columns()
        self.train_file = self._find_file(['train.csv', '*.train'])
        self.test_file = self._find_file(['test.csv', '*.test'])

    def _find_file(self, patterns):
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                return files[0]
        return None

    def _get_columns(self):
        names_file = next(self.data_dir.glob('*.names'), None)
        if names_file:
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
        return df

    def get_train(self):
        if self.train_file is None:
            raise FileNotFoundError("No train file found.")
        return self._clean(self.train_file)

    def get_test(self):
        if self.test_file is None:
            raise FileNotFoundError("No test file found.")
        return self._clean(self.test_file)

    def save_csv(self, output_dir: str):
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df = self.get_train()
        train_df.to_csv(out_dir / 'train.csv', index=False)
        try:
            test_df = self.get_test()
            test_df.to_csv(out_dir / 'test.csv', index=False)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    transformer = DataTransformer("../../data/adult")
    df_train = transformer.get_train()
    df_test = transformer.get_test()
    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape if df_test is not None else "No test file found")
    transformer.save_csv("../../data/adult/csv")
