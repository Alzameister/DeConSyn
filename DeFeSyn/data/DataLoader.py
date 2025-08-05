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


