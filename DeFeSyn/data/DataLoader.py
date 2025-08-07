import logging
from pathlib import Path
from pprint import pprint

import pandas as pd
import yaml


class DatasetLoader:
    """
    Generic dataset loader driven by a YAML manifest in Frictionless Data Table Schema format.
    """
    def __init__(self, manifest_path: str):
        """
        Initialize the loader by reading the YAML manifest and loading each resource.
        :param manifest_path: Path to the YAML manifest file.
        """
        # Load manifest
        manifest_file = Path(manifest_path).expanduser().resolve()
        with open(manifest_file, 'r', encoding='utf-8') as f:
            self.manifest = yaml.safe_load(f)

        self._dataframes = {}
        # Common missing values
        missing_values = self.manifest.get('resources', [])[0] \
            .get('schema', {}).get('missingValues', [])

        # Load each resource
        basepath = manifest_file.parent
        for resource in self.manifest.get('resources', []):
            name = resource['name']
            path = basepath / resource['path']

            # Build pandas.read_csv kwargs
            dialect = resource.get('dialect', {})
            read_kwargs = {
                'filepath_or_buffer': path,
                'header': None if dialect.get('header') is False else 'infer',
                'names': [field['name'] for field in resource['schema']['fields']],
                'na_values': missing_values,
                'skiprows': dialect.get('skipRows', 0),
                'skipinitialspace': True,
            }
            # Handle comment prefix (only first)
            comment_prefixes = dialect.get('commentPrefixes')
            if comment_prefixes:
                read_kwargs['comment'] = comment_prefixes[0]

            # Read the file
            df = pd.read_csv(**read_kwargs)

            # Enforce types and categories
            for field in resource['schema']['fields']:
                col = field['name']
                ftype = field.get('type')
                constraints = field.get('constraints', {})
                if ftype == 'integer':
                    # convert to pandas nullable Int64
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif ftype == 'string':
                    enum = constraints.get('enum')
                    if enum:
                        df[col] = pd.Categorical(df[col], categories=enum)
                    else:
                        # use pandas string dtype
                        df[col] = df[col].astype('string')

            self._dataframes[name] = df

    def resource_names(self):
        """Return all resource names from the manifest."""
        return list(self._dataframes.keys())

    def get(self, name: str) -> pd.DataFrame:
        """
        Get the DataFrame for a specific resource name.
        Raises KeyError if not found.
        """
        return self._dataframes[name]

    def all(self) -> dict:
        """Return a dict of all loaded DataFrames keyed by resource name."""
        return dict(self._dataframes)

    def concat(self, names=None, ignore_index=True) -> pd.DataFrame:
        """
        Concatenate multiple resources into a single DataFrame.
        :param names: List of resource names to combine; if None, uses all.
        :param ignore_index: Whether to ignore original indices.
        """
        keys = names or self.resource_names()
        dfs = [self._dataframes[k] for k in keys]
        return pd.concat(dfs, ignore_index=ignore_index)

    def split(self, n: int, save_path: str) -> tuple[str, str]:
        """
        Split the dataset into n parts and save them as separate resources in the manifest.
        :param n: Number of parts to split into.
        :param save_path: Directory to save the new resources.

        :return: Tuple of paths to the new manifest and the directory where resources are saved.
        """
        if n <= 0:
            raise ValueError("Number of splits must be greater than 0.")

        save_path = Path(save_path).expanduser().resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        manifest_path = save_path / 'manifest.yaml'
        new_manifest = {
            'name': self.manifest.get('name', 'split_dataset'),
            'description': self.manifest.get('description', 'Split dataset resources'),
            'resources': []
        }

        for resource_name, df in list(self._dataframes.items()):
            # Ensure the DataFrame is not empty
            if df.empty:
                continue

            # Shuffle the DataFrame
            # TODO: Consider using a more robust shuffling method if needed
            #df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Split into n parts
            split_size = len(df) // n
            for i in range(n):
                start = i * split_size
                end = (i + 1) * split_size if i < n - 1 else len(df)
                part_df = df.iloc[start:end].copy()

                # Create new resource name
                part_name = f"{resource_name}-part-{i+1}"
                part_path = Path(save_path) / f"{part_name}.{resource_name.split('-')[-1]}"
                part_df.to_csv(part_path, index=False, header=False)
                # Update the manifest with new resources
                new_resource = {
                    'name': part_name,
                    'path': part_path.name,
                    'missingValues': 'NaN',
                    'schema': self.manifest['resources'][0]['schema']
                }
                new_manifest['resources'].append(new_resource)
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(new_manifest, f, allow_unicode=True)
                logging.info(f"Saved {part_name} to {part_path}")
                logging.info(f"Updated manifest saved to {manifest_path}")

        return str(save_path), 'manifest.yaml'

if __name__ == "__main__":
    dataset_metadata_file = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult/adult.yaml"
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


