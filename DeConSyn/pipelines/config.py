from pathlib import Path

import tomli

from DeConSyn.models.tab_ddpm.lib import unpack_config


def load_config(path: Path | str):
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))