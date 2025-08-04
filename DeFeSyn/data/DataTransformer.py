import os
from pathlib import Path
import pandas as pd

class DataTransformer:
    """
    A versatile transformer that reads datasets in various formats (including UCI-style)
    and converts them into pandas DataFrames or saves them as CSV files.

    Supported formats:
    - UCI repository style: folder containing .names, .data, and .test files
    - CSV files
    - JSON files
    - Excel files (.xls, .xlsx)

    Note:
    - For UCI-style datasets, `transform` returns a dict with keys 'train' and 'test'.
      If only a .data file exists, 'test' will be None.
    - `output_csv_dir` is treated as a directory: for single-file sources it writes '<source_name>.csv';
      for UCI it writes 'train.csv' and 'test.csv'.
    """
    def __init__(self):
        pass

    def transform(self, source, output_csv_dir=None, format_hint=None):
        """
        Read the dataset from `source` and return a pandas DataFrame or a dict of DataFrames.
        If `output_csv_dir` is provided, it should be a directory path:
          - For CSV/JSON/Excel, writes a single CSV named '<source_name>.csv'.
          - For UCI-style, writes 'train.csv' and 'test.csv'.
        `format_hint` can be one of ['uci', 'csv', 'json', 'excel'] to force a format.
        """
        # Handle UCI-style datasets separately
        if format_hint == 'uci' or self._looks_like_uci(source):
            dfs = self._read_uci_folder(source)
            if output_csv_dir:
                out_folder = Path(output_csv_dir)
                out_folder.mkdir(parents=True, exist_ok=True)
                dfs['train'].to_csv(out_folder / 'train.csv', index=False)
                if dfs['test'] is not None:
                    dfs['test'].to_csv(out_folder / 'test.csv', index=False)
            return dfs

        # Non-UCI datasets: single DataFrame
        path = Path(source)
        if format_hint == 'csv' or path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        elif format_hint == 'json' or path.suffix.lower() == '.json':
            df = pd.read_json(path)
        elif format_hint == 'excel' or path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format for source: {source}")

        if output_csv_dir:
            out_folder = Path(output_csv_dir)
            out_folder.mkdir(parents=True, exist_ok=True)
            filename = path.stem + '.csv'
            df.to_csv(out_folder / filename, index=False)

        return df

    def _looks_like_uci(self, folder_path):
        """
        Check if folder contains .names and .data files
        """
        folder = Path(folder_path)
        return any(folder.glob('*.names')) and any(folder.glob('*.data'))

    def _parse_names_file(self, names_path):
        """
        Parse a UCI .names file to extract ordered column names.
        """
        columns = []
        with open(names_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('|'):
                    continue
                if ':' in line:
                    name, _ = line.split(':', 1)
                    columns.append(name.strip())
        return columns

    def _read_uci_folder(self, folder_path):
        """
        Read UCI-style dataset from a folder and return a dict with 'train' and 'test' DataFrames.
        Expects files: <dataset>.names, <dataset>.data, optional <dataset>.test
        """
        folder = Path(folder_path)
        # locate .names file
        names_file = next(folder.glob('*.names'), None)
        if not names_file:
            raise FileNotFoundError("No .names file found in UCI folder.")
        columns = self._parse_names_file(names_file)

        # read .data
        data_file = next(folder.glob('*.data'), None)
        if not data_file:
            raise FileNotFoundError("No .data file found in UCI folder.")
        df_train = pd.read_csv(
            data_file,
            header=None,
            names=columns,
            na_values='?',
            skipinitialspace=True
        )

        # read .test if present
        df_test = None
        test_file = next(folder.glob('*.test'), None)
        if test_file:
            df_test = pd.read_csv(
                test_file,
                header=None,
                names=columns,
                na_values='?',
                skipinitialspace=True,
                comment='|'
            )
            # drop any empty first row
            if not df_test.empty and df_test.iloc[0].isnull().all():
                df_test = df_test.drop(0).reset_index(drop=True)

        # cleanup: convert object columns to category and fill missing
        for df in [df_train] + ([df_test] if df_test is not None else []):
            for col in df.select_dtypes(include='object'):
                df[col] = df[col].astype('category')
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

        return {'train': df_train, 'test': df_test}
