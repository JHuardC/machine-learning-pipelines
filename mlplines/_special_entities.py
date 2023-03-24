"""
Supporting objects for pipelines.

created on: Fri 24 Mar 2023

@Author JHuardC
"""
### Imports
from typing import TypeVar
from collections.abc import Iterable
from copy import deepcopy

### UniqueList class

_T = TypeVar('_T')

class UniqueList(list):
    def __init__(self, *args):
        super().__init__(*args)
        if len(self) != len(set(self)):
            raise ValueError(
                f"Arguments contain duplicate elements: {str(args)[1: -1]}"
            )


    def append(self, __object: _T) -> None:
        if __object in self:
            raise ValueError(f"{__object} already exists within list")
        return super().append(__object)

    
    def extend(self, __iterable: Iterable[_T]) -> None:
        if not set(self).isdisjoint(set(deepcopy(__iterable))):
            extension = list(iter(__iterable))
            overlap = [el for el in self if el in extension]
            raise ValueError(
                f"{str(overlap)[1: -1]} elements already exist within list"
            )
        return super().extend(__iterable)


    def __repr__(self) -> str:
        elements = super().__repr__()[1: -1]
        return f"UniqueList({elements})"


######################