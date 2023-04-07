# -*- coding: utf-8 -*-
"""
Component Handler class

Created on: Sun 01 Jan 2023

@author: JHuardC
"""
###################
### Imports
from typing import Final, TypeVar, Union
from mlplines.abc import AbstractComponentHandler, AbstractModellingPipeline
from mlplines.hyper_factory import AbstractHyperparameterHandler
from mlplines.implement_factory import AbstractImplementHandler
from mlplines.saveload_factory import AbstractSaveLoadHandler
from pathlib import Path
from functools import singledispatchmethod
from itertools import zip_longest, starmap
from importlib import import_module
from collections.abc import Iterable

class AliasLookupError(BaseException):
    pass

######################
### Constants
PARENT_MODULE: Final = 'mlplines._builtins.'

###################
### Typing
_Model = TypeVar('_Model')
HyperparameterHandler = AbstractHyperparameterHandler
ImplementHandler = AbstractImplementHandler
SaveLoadHandler = AbstractSaveLoadHandler
ModellingPipeline = TypeVar(
    'ModellingPipeline',
    bound = AbstractModellingPipeline
)
_PathLike = TypeVar('_PathLike', str, Path)
_ComponentHandler = TypeVar('_ComponentHandler', bound = 'ComponentHandler')


######################
### Component Handler
class ComponentHandler(AbstractComponentHandler):
    """
    Loads a model's corresponding implement and hyperparameter handlers.

    Attributes
    ----------
    step_name: str
        Instance attribute to be set on initialization. logs which step 
        in the pipeline the component handler class corresponds to.

    ext_lookup: str or pathlib.Path
        Location of custom hyperparameter and implement handlers built
        for the passed model.

    Properties
    ----------
    hyperparameter_handler: HyperparameterHandler.
        Stores the hyperparameter handler. No set function is given, the
        only intended way to provide a value to this property is through
        the _set_handlers method.

    impliment_handler: ImplimentHandler.
        Stores the impliment handler. No set function is given, the
        only intended way to provide a value to this property is through
        the _set_handlers method.

    saveload_handler: SaveLoadHandler.
        Stores the saving and loading handler. No set function is given,
        the only intended way to provide a value to this property is
        through the _set_handlers method.

    model: object.
        Stores the model used in a specific step by the pipeline. On 
        setting the model _set_handlers is called to set the values of 
        implement_handler, hyperparameter_handler, and saveload_handler.

    Class Methods
    -------------
    load. Returns: ComponentHander.
        Loads a saved ComponentHandler state.

    Methods
    -------
    apply. Returns: Iterable.
        Calls implement handler to process data X on trained model.

    load_model. Returns: _Model
        Loads a model's savestate using saveload_handler.

    save. Returns: None
        Saves model and ComponentHandler wrapper.

    save_model. Returns: None.
        Saves a model's state using saveload_handler.

    train. Returns: None.
        Calls implement handler to train the model on data X.

    train_apply. Returns: Iterable.
        Calls implement handler to train and process data X.

    update_kwargs. Returns: None.
        Calls hyperparameter handler to update the models 
        hyperparameters, the model will be treated as newly instanced
        on update.
    """
    ### Declare attributes
    __hyperparameter_handler: HyperparameterHandler # Instance attr
    __implement_handler: ImplementHandler # Instance attr
    __saveload_handler: SaveLoadHandler # Instance attr

    @property
    def model(self) -> Union[_Model, type[_Model]]:
        return self.__model
    @model.setter
    def model(self, model: Union[_Model, type[_Model]]) -> None:
        self._set_handlers(model)
        self.__model = model

    
    @property
    def hyperparameter_handler(self) -> HyperparameterHandler:
        return self.__hyperparameter_handler


    @property
    def implement_handler(self) -> ImplementHandler:
        return self.__implement_handler


    @property
    def saveload_handler(self) -> SaveLoadHandler:
        return self.__saveload_handler
    
    
    @classmethod
    def load(
        cls: type[_ComponentHandler],
        path: _PathLike,
        reader: callable,
        mode: str = 'r'
    ) -> _ComponentHandler:
        """
        Loads a saved ComponentHandler state.

        Parameters
        ----------
        path: subclass of str or pathlib.Path.
            path to file containing saved state.

        reader: callable.
            A function that can parse the save file's format to python
            data types. pickle.load for a .pkl or .pickle file, for
            example.
        
        mode: str. Default: 'r'.
            Mode in which file should be opened. Accepts only 'r' or
            'rb'.
        """
        if mode not in {'r', 'rb'}:
            raise ValueError(
                f"Incorrect mode argument passed: {mode}.\nMust be 'r' or 'rb'"
            )
        with open(path, mode) as file:
            state: dict = reader(file)
        
        model_state = state.pop('model_state')

        instance: ComponentHandler = cls.__new__(cls)
        instance.__setstate__(state = state)
        instance.load_model(path = model_state)

        return instance


    def __init__(
        self,
        step_name: str,
        model: Union[_Model, type[_Model]],
        init_kwargs: Union[dict, None] = None,
        ext_lookup: Union[_PathLike, None] = None
    ) -> None:
        """
        Loads a model's corresponding implement and hyperparameter 
        handlers.

        Parameters
        ----------
        step_name: str
            Instance attribute to be set on initialization. logs which 
            step in the pipeline the component handler class corresponds
            to.

        model: object.
            Stores the model used in a specific step by the pipeline. On
            setting the model _set_handlers is called to set the values 
            of implement_handler and hyperparameter_handler.

        init_kwargs: dict.
            Kwargs to be passed to the model on initialization.

        ext_lookup: str or pathlib.Path
            Optional argument. Location of custom hyperparameter and
            implement handlers built for the model passed.
        """
        self.step_name = step_name

        if isinstance(ext_lookup, str):
            self.ext_lookup = Path(ext_lookup)
        else:
            self.ext_lookup = ext_lookup

        self.model = model
        if init_kwargs:
            self.update_kwargs(pipeline_external_kwargs = init_kwargs)


    @singledispatchmethod
    def _set_handlers(self, model: object) -> None:
        """
        Retrieves a model's corresponding hyperparameter and implement
        handlers.
        """
        # called when a model instance is passed
        model_class = type(model) # get type
        self._set_handlers(model_class) # re-try

    @_set_handlers.register(type)
    def _(self, model) -> None:

        bases = [
            AbstractHyperparameterHandler,
            AbstractImplementHandler,
            AbstractSaveLoadHandler
        ]

        hyper, imp, svld = tuple(
            starmap(
                self._get_child, zip_longest([], bases, fillvalue = model)
            )
        )
        
        self.__hyperparameter_handler = hyper(step_name = self.step_name)
        self.__implement_handler = imp(step_name = self.step_name)
        self.__saveload_handler = svld(step_name = self.step_name)


    def _get_child(
        self,
        model: type[_Model],
        base_cls: type[
            Union[AbstractHyperparameterHandler, AbstractImplementHandler]
        ]
    ) -> type[Union[HyperparameterHandler, ImplementHandler]]:
        """
        Retrieve a model's corresponding handler class.

        Parameters
        ----------
        model: _Model (uninitialized).
            Chilren of base_cls are checked against the argument passed
            to find the model's corresponding Handler class.

        base_cls: AbstractHyperparameterHandler\AbstractImplementHandler
            All Hyperparameter\Implement Handler classes should inherit
            one of these corresponding classes, which in turn allows
            this method to search through the Root classes for them.

        Returns
        -------
        HyperparameterHandler or ImplementHandler
        """
        # Attempt 1 -  Handler has already been loaded into global space
        handler = list(base_cls.match_cls_progeny(model = model))

        if len(handler) > 1:
            # A model should have a single handler class
            self.__raise_error(model = model, candidates = handler)

        elif len(handler) == 1:
            return handler.pop()

        elif (root_name := model.__module__.split('.')[0]) == '__main__':
            # Models built in main file should have their handlers built
            # in main too.
            self.__raise_error(model = model, candidates = handler)

        else:
            root_name += '_extensions'
        
        # Attempt 2 - Search for handlers in ext_lookup using root_name
        if isinstance(self.ext_lookup, Path):
            ext_lookup = self.ext_lookup.joinpath(f'{root_name}.py')
            if ext_lookup.is_file():

                # load contents of root_name to globals
                with open(ext_lookup, 'r') as file:
                    exec(file.read(), dict(__name__ = ''))

                # load handlers
                handler = list(base_cls.match_cls_progeny(model = model))
                if len(handler) != 1:
                    self.__raise_error(model = model, candidates = handler)
                else:
                    return handler.pop()

        # Attempt 3 - Search for builtin handlers
        try:
            imp = import_module(PARENT_MODULE + root_name)
        except(ModuleNotFoundError):
            raise AliasLookupError('No Handler found for the given model.')

        # load handlers
        handler = list(base_cls.match_cls_progeny(model = model))
        if len(handler) != 1:
            self.__raise_error(model = model, candidates = handler)
        else:
            return handler.pop()


    def __raise_error(
        self,
        model: type[_Model],
        candidates: list[type[Union[HyperparameterHandler, ImplementHandler]]]
    ) -> None:
        """
        Determine the error message to raise.

        Called when _get_child method returns an incorrect number of 
        handler classes - i.e. anything other than 1.

        Parameters
        ----------
        candidates: list of Hyperparameter\Implement Handlers.
            Classes matched to a given model. Relationship from a 
            model to both Handler types should be many-to-one.
        """
        if len(candidates) > 1:
            error_msg = [
                f"Model passed: {model} - retreived more than one ",
                "handler class:\n\n"
            ]
            error_msg.extend(
                [
                    f"{el.get_match_description()}\n\n" for el in candidates
                ]
            )
            error_msg = ''.join(error_msg)

        elif len(candidates) == 0:
            error_msg = "No associated handler class for the model given"

        else:
            error_msg = f"len(candidates) = {len(candidates)}. Unknown error."

        raise AliasLookupError(error_msg)
    

    def get_handler_state(self) -> dict:
        """
        Gets the states for self and handler attributes, but not model's
        state.
        """
        if isinstance(self.model, type):
            model = self.model.__name__
        else:
            model: str = self.model.__class__.__name__
        
        state = dict(
            step_name = self.step_name,
            ext_lookup = str(self.ext_lookup),
            hyper_state = self.hyperparameter_handler.__getstate__(),
            implement_state = self.implement_handler.__getstate__(),
            model_module = self.model.__module__,
            model_class =  model
        )

        return state
    

    def __getstate__(self) -> dict:
        state = self.get_handler_state()
        state.update(
            model_state = self.saveload_handler.get_model_state(self.model)
        )
        return state
    

    def _set_handler_state(self, state: dict) -> None:
        """
        Sets values for self and handler attributes, but not for the
        model.
        """
        model: type[_Model] = getattr(
            import_module(state.pop('model_module')),
            state.pop('model_class')
        )
        hyper_state = state.pop('hyper_state')
        implement_state = state.pop('implement_state')

        self.__dict__.update(state)
        self._set_handlers(model)
        self.hyperparameter_handler.__setstate__(state = hyper_state)
        self.implement_handler.__setstate__(state = implement_state)
        self.__model = model


    def __setstate__(self, state: dict) -> None:
        model_state = state.pop('model_state', False)

        self._set_handler_state(state = state)

        if model_state:
            self.__model = self.saveload_handler.set_model_state(
                self.__model,
                state['model_state']
            )


    def load_model(
        self,
        path: _PathLike,
        **kwargs
    ) -> None:
        """
        Loads a saved model state. Assumes model's module contains it's
        own save/load functionality.

        Parameters
        ----------
        path: subclass of str or pathlib.Path.
            path to file containing model's saved state.
        """
        self.__model = self.saveload_handler.load(self.__model, path, **kwargs)


    def save_model(
        self,
        to: _PathLike,
        **model_kwargs
    ) -> None:
        """
        Saves model state.

        Parameters
        ----------
        to: str or pathlib.Path
            File location the model will be saved to.
        """
        self.saveload_handler.save(to, self.model, **model_kwargs)


    def save(
        self,
        to: _PathLike,
        writer: callable,
        mode: str = 'w',
        model_path_suffix: str = '',
        **model_save_kwargs
    ) -> None:
        """
        Saves ComponentHandler and model's state.
        
        The model is saved to a seperate file/collection of files from
        the rest of the ComponentHandler.

        Parameters
        ----------
        to: str or pathlib.Path
            File location the model will be saved to.

        writer: callable.
            A function that can parse python data types to the save
            file's format. pickle.dump for a .pkl or .pickle file, for
            example.
        
        mode: str. Default: 'w'.
            Mode in which file should be opened. Accepts only 'w' or
            'wb'.

        model_path_suffix: str. Default: ''.
            Optional argument dependending on the model's save
            functionality. In some instances, a package may infer save
            type by the destination filename's suffix.

        Kwargs
        ------
        Any kwargs to be passed to the models in-house save
        functionality.
        """
        if mode not in {'w', 'wb'}:
            raise ValueError(
                f"Incorrect mode argument passed: {mode}.\nMust be 'w' or 'wb'"
            )

        model_path = self.saveload_handler.get_model_path(
            to,
            self.model,
            model_path_suffix
        )
        self.save_model(model_path, **model_save_kwargs)

        self_state = self.get_handler_state()
        self_state.update(model_state = model_path)

        with open(to, mode) as f:
            writer(self_state, f)


    def update_kwargs(
        self,
        pipeline_external_kwargs: dict,
        pipeline_env_kwargs: Union[dict, None] = None
    ) -> None:
        """
        Updates the hyperparameters of the model.
        """
        if pipeline_env_kwargs == None:
            pipeline_env_kwargs = dict()

        self.__model = self.__hyperparameter_handler(
            model = self.model,
            pipeline_external_kwargs = pipeline_external_kwargs,
            pipeline_env_kwargs = pipeline_env_kwargs
        )


    def train(
        self,
        X: Iterable,
        y: Union[Iterable, None] = None,
        pipeline: Union[ModellingPipeline, None] = None
    ) -> None:
        """
        Trains the model.
        """
        self.__model = self.__implement_handler.train(
            pipeline = pipeline, model = self.model, X = X
        )


    def apply(
        self,
        X: Iterable
    ) -> Iterable:
        """
        Applies the trained model to the data.
        """
        return self.__implement_handler.apply(model = self.model, X = X)


    def train_apply(
        self,
        X: Iterable,
        y: Union[Iterable, None] = None,
        pipeline: Union[ModellingPipeline, None] = None
    ) -> Iterable:
        """
        Trains the model and applies it to the training data.
        """
        output, self.__model = self.__implement_handler.train_apply(
            pipeline = pipeline, model = self.model, X = X, y = y
        )

        return output