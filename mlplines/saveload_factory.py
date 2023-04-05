# UTF-8 Encoding
"""
Save and Load Handling Template CLasses.

Inherit one of these classes and override it's abstract methods to 
create a bespoke saveload handler for a given model.

Created on: Tue 28 Mar 2023

@author: JHuardC
"""
### Imports
from typing import TypeVar
from pathlib import Path
from mlplines.abc import BaseHandler, SaveLoadHandlerMixin

### Types
_Model = TypeVar('_Model')
_PathLike = TypeVar('_PathLike', str, Path)

### Classes
class AbstractSaveLoadHandler(BaseHandler, SaveLoadHandlerMixin):
    """
    Template for all save-load Handler classes.

    Root class for all concrete save-load handler classes: The progeny
    of this class will be used by ComponentHandler.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Methods
    ----------------
    get_model_state. Returns: _Picklable
        Retrieves the model's state.

    set_model_state. Returns: _Model.
        Calls any internal load function used by the model.

    save. Returns: PathLike
        Calls any internal save function used by the model.

    load. Returns: _Model.
        Retrieves a saved state and calls any internal load function
        used by the model.
    
    Methods
    -------
    get_model_path. Returns: pathlib.Path
        Generates a separate path name for the model to be saved to
        using the save/load functionality recommended for the module.
    
    get_match_description. Returns: str.
        maps to _check_model's method of the same name. Describes the 
        conditions required for the Hyperparameter/Implement Handler 
        to be called

    has_check_model. Returns: bool.
        Checks whether this class has a _check_model class attribute.

    match_cls_progeny. Returns: Iterable containing class or a subclass.
        Recursive class method. Returns correct handler class for the 
        given model.
    """
    def get_model_path(
            self,
            component_handler_path: _PathLike,
            model: _Model,
            path_suffix: str
        ) -> Path:
        """
        Generates a separate path name for the model to be saved to
        using the save/load functionality recommended for the module.
        """
        chp = component_handler_path

        model_file_name = f'gensim_{model.__class__.__name__}{path_suffix}'
        if isinstance(chp, Path):
            model_path = chp.parent.joinpath(model_file_name)
        else:
            model_path = Path(chp).absolute().parent.joinpath(model_file_name)

        return model_path