"""
Key constants used by test scripts.

created on: Fri 07 Apr 2023

created by: JHuardC
"""
### Imports
from typing import Final
from pathlib import Path
from ._dir_utils import go_to_ancestor

### Constants
DATA_PATH: Final[Path] = go_to_ancestor(Path()).joinpath(
    'tests',
    'test_data',
    'petitions_sample.pqt'
)

OUTPUT_Path: Final[Path] = go_to_ancestor(Path()).joinpath(
    'tests',
    'test_outputs'
)
