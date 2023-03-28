# -*- coding: utf-8 -*-
"""
Hyperparameter Handling Template CLasses.

Inherit one of these classes and override it's abstract methods to 
create a bespoke hyperparameter handler for a given model.

Created on: Sat 31 Dec 2022

@author: JHuardC
"""
###################
### Imports
from typing import TypeVar, Union
from mlplines.abc import BaseHandler, HyperparameterHandlerMixin
from abc import abstractmethod

###################
### Type Variables/Aliases
_Model = TypeVar('_Model')

###################
### Abstract Hyperparameter handler templates
class AbstractHyperparameterHandler(BaseHandler, HyperparameterHandlerMixin):
    """
    Template for all Hyperparameter Handler classes.

    Root class for all concrete hyperparameter handler classes: The 
    progeny of this class will be used by ComponentHandler.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    --------------------
    _relevant_pipeline_env_keys: tuple[str, ...].
        Some required Hyperparameter values may be derived from 
        previous pipeline steps. In such cases these arguments should be
        retreived using the keys stored in this property.

    Abstract Methods
    ----------------
    _call_with_model_object. Returns: Model.
        This method is called when the model requires initialization.

    _call_with_model_instance. Returns: Model.
        This method is called when the model has been initialized, but 
        updates to it's hyperparameters are required.

    _call_without_hyperparameters. Returns: Model.
        When no hyperparameter values have been specified. Generally 
        implies do nothing but return the model argument as is.
    
    Methods
    -------
    __call__:
        Once initialized, hyperparameter handler classes are called 
        as a functions to update a models' hyperparameters. The call 
        method determines when to call one of the methods:

            _call_with_model_object
            _call_with_model_instance
            _call_without_hyperparameters
    
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
    @property
    @abstractmethod
    def _relevant_pipeline_env_keys(self) -> tuple[str, ...]:
        pass


    @abstractmethod
    def _call_with_model_object(
        self,
        model: type[_Model],
        **kwargs
    ) -> _Model:
        pass


    @abstractmethod
    def _call_with_model_instance(
        self,
        model: _Model,
        **kwargs
    ) -> _Model:
        pass


    @abstractmethod
    def _call_without_hyperparameters(
        self, model: Union[_Model, type[_Model]]
    ) -> _Model:
        pass


    def _get_relevant_env_kwargs(self, env_kwargs: dict) -> dict:
        return dict(
            (k, env_kwargs[k]) for k in self._relevant_pipeline_env_keys
            if k in env_kwargs
        )


    def __call__(
        self,
        model: Union[_Model, type[_Model]],
        pipeline_external_kwargs: dict,
        pipeline_env_kwargs: dict
    ) -> _Model:
        # TODO: check and log warning for shaired keys
        kwargs = self._get_relevant_env_kwargs(pipeline_env_kwargs)
        kwargs = {**kwargs, **pipeline_external_kwargs}

        if not kwargs:
            return self._call_without_hyperparameters(model = model)

        elif isinstance(model, type):
            return self._call_with_model_object(model, **kwargs)

        else:
            return self._call_with_model_instance(model, **kwargs)


###################
### Partially implemented Hyperparameter handler templates
class SimpleHyperparameterHandler(AbstractHyperparameterHandler):
    """
    Models that can be updated directly, by passing new values to their 
    attributes.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    --------------------
    _relevant_pipeline_env_keys: tuple[str, ...].
        Some required Hyperparameter values may be derived from 
        previous pipeline steps. In such cases these arguments should be
        retreived using the keys stored in this property.
    
    Methods
    -------
    __call__:
        Once initialized, hyperparameter handler classes are called 
        as a functions to update a models' hyperparameters.
    
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
    def _call_without_hyperparameters(
        self, model: Union[_Model, type[_Model]]
    ) -> _Model:
        if isinstance(model, type):
            raise ValueError('No arguments passed to initialize model.')
        else:
            return model

    def _call_with_model_object(
        self,
        model: type[_Model],
        **kwargs
    ) -> _Model:
        return model(**kwargs)


    def _call_with_model_instance(
        self,
        model: _Model,
        **kwargs
    ) -> _Model:
        for attr, val in kwargs.items():
            setattr(model, attr, val)

        return model


class InitializeToUpdateHandler(AbstractHyperparameterHandler):
    """
    Template for models that perform some calculations on initialization
    
    Such model instances cannot have their attributes updated with new 
    values, therefore all such models should have their initial kwargs 
    recorded and any updates to the model's hyperparameters will be 
    passed to a new instance of the model on initialization alongside 
    the initial arguments.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    --------------------
    _relevant_pipeline_env_keys: tuple[str, ...].
        Some required Hyperparameter values may be derived from 
        previous pipeline steps. In such cases these arguments should be
        retreived using the keys stored in this property.
    
    Methods
    -------
    __call__:
        Once initialized, hyperparameter handler classes are called 
        as a functions to update a models' hyperparameters.
    
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
    def _call_without_hyperparameters(
        self, model: Union[_Model, type[_Model]]
    ) -> type[_Model]:
        if isinstance(model, type):
            raise ValueError('No arguments passed to initialize model.')
        else:
            return model


    def _call_with_model_object(
        self,
        model: type[_Model],
        **kwargs
    ) -> _Model:
        """
        Called to initialize the model. Initial kwargs are stored.

        In theory, this method should only be called once for any 
        instance of this class.
        """
        self.initial_kwargs = kwargs.copy()
        return model(**kwargs)


    def _call_with_model_instance(
        self,
        model: _Model,
        **kwargs
    ) -> _Model:
        """
        Called when hyperparameter updates are being passed to a model.
        """
        model = model.__class__
        return model(**{**self.initial_kwargs, **kwargs})


class TrainOnInitializationHandler(AbstractHyperparameterHandler):
    """
    Template for models that expect data to train on initialization. 
    
    Such models cannot have their instances updated with new kwargs, 
    therefore all such models should have their initial arguments 
    recorded and any updates to the model's hyperparameters will be 
    passed to a new instance of the model on initialization alongside 
    the recorded initial arguments.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    --------------------
    data_kwarg: str.
        Parameter name in Model's initialization method for training 
        data.
        
    _relevant_pipeline_env_keys: tuple[str, ...].
        Some required Hyperparameter values may be derived from 
        previous pipeline steps. In such cases these arguments should be
        retreived using the keys stored in this property.
    
    Methods
    -------
    __call__:
        Once initialized, hyperparameter handler classes are called 
        as a functions to update a models' hyperparameters.
    
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
    @property
    @abstractmethod
    def data_kwarg(self) -> str:
        """
        Parameter name for training data in Model's initialization.
        """
        pass

    def _get_stored_kwargs(self) -> dict:
        return getattr(
            self,
            f'_{self.__class__.__name__}__updated_kwargs',
            getattr(self, 'initial_kwargs')
        )


    def _call_without_hyperparameters(
        self, model: Union[_Model, type[_Model]]
    ) -> type[_Model]:
        if isinstance(model, type):
            return model
        else:
            return model.__class__


    def _call_with_model_object(
        self,
        model: type[_Model],
        **kwargs
    ) -> Union[_Model, type[_Model]]:
        """
        Called when either a model is trained, or to record initial 
        kwargs, or if updates are passed to a trained model more than 
        once.
        """
        if self.data_kwarg in kwargs: # called for training
            return model(**{**self._get_stored_kwargs(), **kwargs})

        elif not hasattr(self, 'initial_kwargs'):
            self.initial_kwargs = kwargs.copy()
            return model

        else: # update called more than once
            self.__updated_kwargs = {**self.initial_kwargs, **kwargs}
            return model


    def _call_with_model_instance(
        self,
        model: _Model,
        **kwargs
    ) -> type[_Model]:
        """
        Called when updates are being passed to a trained model.
        """
        self.__updated_kwargs = {**self.initial_kwargs, **kwargs}
        return model.__class__