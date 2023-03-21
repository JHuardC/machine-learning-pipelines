# UTF-8 Encoding
"""
Check Model Template Classes.

Inherit one of these classes and override it's abstract methods to 
create a bespoke Check Model class that only returns True for specific
models.

Created on: Wed 01 Feb 2023

@author: Joe Huard
"""
### Imports
from typing import TypeVar, Union
from topic_modelling_pipelines.pipelines.abc import\
    AbstractCheckModel, CheckModelTrue

from topic_modelling_pipelines.pipelines.hyper_factory import\
    AbstractHyperparameterHandler

from topic_modelling_pipelines.pipelines.implement_factory import\
    AbstractImplementHandler

### Typing
_Model = TypeVar('_Model')
_Handler = TypeVar(
    '_Handler',
    bound = Union[AbstractHyperparameterHandler, AbstractImplementHandler]
)

### Template classes
class SubclassOfKey(AbstractCheckModel):
    """
    Checks model is a subclass of the provided key.

    Abstract Methods
    ----------------
    get_key. Returns: type or tuple of types.
        Class Method. Corresponding models will be subclasses of the 
        type returned.
    """

    @classmethod
    def get_match_description(
        cls, realization_cls: type[_Handler]
    ) -> str:
        output = ''.join(
            [
                f"{realization_cls} is matched if the model passed is a ",
                f"subclass of one of the following: {cls.get_key()}."
            ]
        )
        return output

    @classmethod
    def check_model(cls, model: _Model) -> bool:
        return issubclass(model, cls.get_key())