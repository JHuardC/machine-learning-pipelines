# -*- coding: utf-8 -*-
""" Utitity classes and functions test scripts.

Created on: Sat 1 Apr 2023

@author: JHuardC
"""

from typing import Any, Final
import pathlib as pl

### Constant
PROJECT_NAME: Final[str] = 'machine-learning-pipelines'

### Pathing Functions
def _get_path_TypeError_text(path: Any) -> str:
    """Error message to explain path arg is not a pathlib.Path type."""
    arg_type = type(path).__name__
    return f'{path = } is of type {arg_type}. Not of type pathlib.Path'


def _get_ancestor_TypeError_text(ancestor: Any) -> str:
    """Error message to explain ancestor arg is not a 'str' type."""
    arg_type = type(ancestor).__name__
    return f'{ancestor = } is of type {arg_type}. Not of type str'


def go_to_ancestor(path: pl.Path, ancestor: str = PROJECT_NAME) -> pl.Path:
    """Returns the path of an ancestor to the specified path.
    
    Parameters
    ``````````
    path [pathlib.Path]
        Specified path to navigate.

    ancestor [str]
        Name of directory part to navigate to.
        
    Returns
    ```````
    Ancestor path [pathlib.Path]
    """
    if isinstance(path, pl.Path):
        path_parts = path.parts
    else:
        raise TypeError(_get_path_TypeError_text(path))
        
    if isinstance(ancestor, str):
        try:
            ancestor_index = path_parts.index(ancestor)
        except ValueError as verr:
            verr.args = (f'{ancestor = } is not present within {path = }',)
            raise verr
    else:
        raise TypeError(_get_ancestor_TypeError_text(ancestor))
        
    ancestor_parts = path_parts[:ancestor_index + 1]
    
    return pl.Path(*ancestor_parts)