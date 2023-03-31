# -*- coding: utf-8 -*-
""" 
Abstract Base Classes for modelling pipelines and component handlers

The Abstract classes/methods here have been designed to blueprint how 
the modelling pipeline will work, whilst providing as much flexibility 
as possible for future development. To provide flexibility for 
development, a minimal number of class methods have been introduced and 
of those a minimal number have been made concrete.

Created on: Thu 01 Sep 2022

@author: JHuardC
"""
### Imports
from abc import abstractmethod, ABC
from typing import TypeVar, Literal, Union, ClassVar
from collections.abc import Iterable
from itertools import chain, repeat
from pathlib import Path
from mlplines._special_entities import UniqueList

######################
### Define Types
_spec_keys = Literal['step_name', 'model', 'init_kwargs']
_step_name = str
_Model = TypeVar('_Model')
_init_kwargs = dict

_ModelSpec = dict[
    _spec_keys,
    Union[_step_name, _Model, type[_Model], _init_kwargs]
]

_ModelCriteria = TypeVar('_ModelCriteria')

_CheckModel = TypeVar('_CheckModel', bound = 'AbstractCheckModel')
_RootHandler = TypeVar('_RootHandler', bound = 'RootMixin')

_Picklable = TypeVar('_Picklable')

PathLike = TypeVar('PathLike', Path, str)

######################
### Abstract check model class
class AbstractCheckModel:
    """
    Provides check_model method to Hyperparameter Handler and Implement
    Handler classes.

    Every Hyperparameter/Implement/SaveLoad Handler class should contain
    a child of AbstractCheckModel as a class attribute to check whether
    the handler class is the correct class to use with the passed model.

    Abstract Methods
    ----------------
    get_key. Returns: _ModelCriteria (criteria to match a model.)
        check_model classmethod will check a model against the criteria
        returned by this method.
    
    get_match_description. Returns: str.
        Explains the context for when the class will match to a model.
    
    check_model. Returns: bool.
        Confirms whether this class is the correct handler class for the
        component model.
    """
    @classmethod
    @abstractmethod
    def get_key(cls: type[_CheckModel]) -> _ModelCriteria:
        """
        class method check_model will match the component model against 
        the criteria returned by this method.
        """
        pass

    @classmethod
    @abstractmethod
    def get_match_description(
        cls: type[_CheckModel], realization_cls: type[_RootHandler]
    ) -> str:
        """
        Describes when the handler class will match to a model.

        Parameters
        ----------
        realization_cls: type[_RootHandler].
            The Hyperparameter/Implement/SaveLoad handler class that is
            calling this method.
        """
        pass
    
    @classmethod
    @abstractmethod
    def check_model(cls: type[_CheckModel], model: _Model) -> bool:
        """
        Confirm whether the realization class calling this method is the
        correct handler class for the given model.

        Parameters
        ----------
        model: _Model.
            Model passed to a Component Handler class in a modelling 
            pipeline.

        Returns
        -------
        bool: True if the Hyperparameter/Implement/SaveLoad handler is the 
        correct Handler class to the corresponding component model.
        """
        pass

################################
### Concrete Check Model class

class CheckModelTrue(AbstractCheckModel):
    """
    Always returns True for check_model method.
    """
    @classmethod
    def get_key(cls: type[_CheckModel]) -> Literal[True]:
        return True

    @classmethod
    def get_match_description(
        cls: type[_CheckModel],
        realization_cls: type[_RootHandler]
    ) -> str:
        return f"{realization_cls}: Will always return True."

    @classmethod
    def check_model(cls: type[_CheckModel], model: _Model) -> Literal[True]:
        return True

######################
### Mixin containing the methods used to search for correct model handlers
class RootMixin:
    """
    Adds methods to search for correct Hyperparameter/Implement/SaveLoad
    handlers.

    The correct Hyperparameter/Implement/SaveLoad Handlers for a given
    model are  found by tracing through the inheritance tree of a Root
    Handler mixin class.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Class Methods
    -------------
    get_match_description. Returns: str.
        maps to __check_model's method of the same name. Describes the
        conditions required for the Hyperparameter/Implement/SaveLoad
        Handler to be returned.

    match_cls_progeny. Returns: Iterable containing class or a subclass.
        Recursive class method. Returns correct handler class for the
        given model.
    """
    __check_model: ClassVar[type[AbstractCheckModel]]

    @classmethod
    def match_cls_progeny(
        cls: type[_RootHandler],
        model: _Model
    ) -> Iterable[type[_RootHandler]]:
        """
        Returns correct handler class for the given model.

        Correct handler class will be returned in an iterable.

        Only one handler class should ever be returned.
        """
        try:
            check_model: AbstractCheckModel = getattr(
                cls, f'_{cls.__name__}__check_model'
            )
        except AttributeError:
            
            # TODO: add warning of no class attribute having been 
            # assigned?
            children = cls.__subclasses__()
            if children:
                return chain.from_iterable(
                    child.match_cls_progeny(model) for child in children
                )
            else:
                return repeat(None, 0)

        except Exception as wild:
            raise wild

        else:

            if check_model.check_model(model = model):
                
                children = cls.__subclasses__()
                if children:
                    return chain.from_iterable(
                        child.match_cls_progeny(model) for child in children
                    )
                else:
                    return repeat(cls, 1)

            else:
                return repeat(None, 0)

    @classmethod
    def get_match_description(cls: type[_RootHandler]) -> str:
        
        try:
            check_model: AbstractCheckModel = getattr(
                cls, f'_{cls.__name__}__check_model'
            )
        except AttributeError:
            
            return f"{cls}: has no class attribute: '__check_model'"

        except Exception as wild:
            raise wild

        else:
            
            return check_model.get_match_description(realization_cls = cls)


#########################################
### Hyperparameter and Implement Handlers
class BaseHandler:
    """
    Provides shared step_name property and init method for all Handler
    classes
    """
    @property
    def step_name(self) -> _step_name:
        return self.__step_name

    def __init__(self, step_name: _step_name) -> None:
        self.__step_name = step_name
        

### Hyperparameter Handler
class HyperparameterHandlerMixin(RootMixin):
    """
    Adds key methods for hyperparameter handlers.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Methods
    ----------------
    __call__:
        An initialized subclass will be called as a function to update
        a models' hyperparameters.

    __getstate__. Returns: dict.
        Gets key instance attributes.
    
    __setstate__. Returns: None.
        Sets key instance attributes.

    Methods
    -------
    get_match_description. Returns: str.
        maps to _check_model's method of the same name. Describes the
        conditions required for the Hyperparameter Handler to be called.

    has_check_model. Returns: bool.
        Checks whether this class has a _check_model class attribute.

    match_cls_progeny. Returns: Iterable containing class or a subclass.
        Recursive class method. Returns correct handler class for the 
        given model.
    """
    @abstractmethod
    def __getstate__(self) -> dict:
        pass

    @abstractmethod
    def __setstate__(self, state: dict) -> None:
        pass

    @abstractmethod
    def __call__(self, model: _Model) -> _Model:
        pass

HyperparameterHandler = TypeVar(
    'HyperparameterHandler',
    bound = '_HyperparameterHandler'
)
class _HyperparameterHandler(ABC, HyperparameterHandlerMixin):
    """
    Used for type annotations in methods.
    """
    @classmethod
    def __subclasshook__(
        cls: type[HyperparameterHandler],
        C: type
    ) -> bool:
        return {BaseHandler, HyperparameterHandlerMixin}.issubset(set(C.mro()))


### Implement Handler
class ImplementHandlerMixin(RootMixin):
    """
    Adds key methods for Implement handlers.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Methods
    ----------------
    train. Returns: None.
        Trains the model on data X.

    apply. Returns: Iterable.
        Processes data X on trained model.

    train_apply. Returns: Iterable.
        Trains and processes data X.

    __getstate__. Returns: dict.
        Gets key instance attributes.
    
    __setstate__. Returns: None.
        Sets key instance attributes.

    Methods
    -------
    get_match_description. Returns: str.
        maps to _check_model's method of the same name. Describes the
        conditions required for the Implement Handler to be called

    has_check_model. Returns: bool.
        Checks whether this class has a _check_model class attribute.

    match_cls_progeny. Returns: Iterable containing class or a subclass.
        Recursive class method. Returns correct handler class for the
        given model.
    """
    @abstractmethod
    def __getstate__(self) -> dict:
        pass

    @abstractmethod
    def __setstate__(self, state: dict) -> None:
        pass

    @abstractmethod
    def train(
        self,
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> None:
        pass

    @abstractmethod
    def apply(
        self,
        model: _Model,
        X: Iterable
    ) -> Iterable:
        pass

    @abstractmethod
    def train_apply(
        self,
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> Iterable:
        pass

ImplementHandler = TypeVar(
    'ImplementHandler',
    bound = '_ImplementHandler'
)
class _ImplementHandler(ABC, ImplementHandlerMixin):
    """
    Used for type annotations in methods.
    """
    @classmethod
    def __subclasshook__(
        cls: type[ImplementHandler],
        C: type
    ) -> bool:
        return {BaseHandler, ImplementHandlerMixin}.issubset(set(C.mro()))


### Save Load Handler
class SaveLoadHandlerMixin(RootMixin):
    """
    Adds key methods for SaveLoad handlers.

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
    get_match_description. Returns: str.
        maps to _check_model's method of the same name. Describes the
        conditions required for the SaveLoad Handler to be called.

    has_check_model. Returns: bool.
        Checks whether this class has a _check_model class attribute.

    match_cls_progeny. Returns: Iterable containing class or a subclass.
        Recursive class method. Returns correct handler class for the
        given model.
    """
    @abstractmethod
    def get_model_state(
        self,
        model: _Model
    ) -> _Picklable:
        pass

    @abstractmethod
    def set_model_state(
        self,
        model: type[_Model],
        state: dict
    ) -> _Model:
        pass

    @abstractmethod
    def save(
        self,
        to: PathLike,
        model: _Model
    ) -> PathLike:
        pass

    @abstractmethod
    def load(
        self,
        model: type[_Model],
        path: PathLike
    ) -> _Model:
        pass

SaveLoadHandler = TypeVar(
    'SaveLoadHandler',
    bound = '_SaveLoadHandler'
)
class _SaveLoadHandler(ABC, ImplementHandlerMixin):
    """
    Used for type annotations in methods.
    """
    @classmethod
    def __subclasshook__(
        cls: type[SaveLoadHandler],
        C: type
    ) -> bool:
        return {BaseHandler, ImplementHandlerMixin}.issubset(set(C.mro()))

######################
### Abstract Component Handler
_AbstractComponentHandler = TypeVar(
    '_AbstractComponentHandler',
    bound = 'AbstractComponentHandler'
)
class AbstractComponentHandler:
    """
    Component handler classes load the corresponding implement and 
    hyperparameter handler classes for a component used as a step in a 
    Modelling Pipeline.

    Required Attributes
    -------------------
    step_name: str
        Instance attribute to be set on initialization. logs which step 
        in the pipeline the component handler class corresponds to.

    Abstract Properties
    -------------------
    hyperparameter_handler: _HyperparameterHandler.
        Stores the hyperparameter handler. No set function is given, the
        only intended way to provide a value to this property is through
        the _set_handlers method.

    impliment_handler: _ImplimentHandler.
        Stores the impliment handler. No set function is given, the
        only intended way to provide a value to this property is through
        the _set_handlers method.

    saveload_handler: _SaveLoadHandler.
        Stores the saving and loading handler. No set function is given,
        the only intended way to provide a value to this property is
        through the _set_handlers method.

    Abstract Class Methods
    ----------------------
    load. Returns: _AbstractComponentHander.
        Loads a saved ComponentHandler state.

    Abstract Methods
    ----------------
    _set_handlers. Returns: None.
        Retreives the HyperparameterHandler, ImplementHandler, and
        SaveLoadHandler classes corresponding to the model passed; these
        classes are stored in the properties noted above.

    save. Returns: _Picklable
        Saves model with ComponentHandler wrapper.

    update_kwargs. Returns: None.
        Calls hyperparameter handler to update the models
        hyperparameters, the model will be treated as newly instanced
        on update.

    train. Returns: None.
        Calls implement handler to train the model on data X.

    apply. Returns: Iterable.
        Calls implement handler to process data X on trained model.

    train_apply. Returns: Iterable.
        Calls implement handler to train and process data X.
    """

    @classmethod
    @abstractmethod
    def load(
        cls: type[_AbstractComponentHandler],
        Path: PathLike
    ) -> _AbstractComponentHandler:
        pass

    step_name: _step_name

    @property
    def model(self) -> Union[_Model, type[_Model]]:
        return self.__model
    @model.setter
    def model(self, model: Union[_Model, type[_Model]]) -> None:
        self._set_handlers(model)
        self.__model = model

    @property
    @abstractmethod
    def hyperparameter_handler(self) -> _HyperparameterHandler:
        pass

    @property
    @abstractmethod
    def implement_handler(self) -> _ImplementHandler:
        pass

    @property
    @abstractmethod
    def saveload_handler(self) -> _SaveLoadHandler:
        pass

    @abstractmethod
    def _set_handlers(
        self, 
        model: Union[_Model, type[_Model]]
    ) -> None:
        pass

    @abstractmethod
    def save(
        self,
        Path: PathLike
    ) -> None:
        pass

    @abstractmethod
    def update_kwargs(self) -> None:
        pass

    @abstractmethod
    def train(self, X: Iterable, y: Union[Iterable, None] = None) -> None:
        pass

    @abstractmethod
    def apply(self, X: Iterable) -> Iterable:
        pass

    @abstractmethod
    def train_apply(
        self,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> Iterable:
        pass


### type alias
_ComponentHandler = AbstractComponentHandler

######################
### Abstract Modelling Pipeline
_PipelineComponent = Union[_ComponentHandler, _ModelSpec, type[_Model]]

class AbstractModellingPipeline:
    """
    Modelling pipeline classes are designed to coordinate the processing
    of data that requires squential use of multiple objects.

    E.g. Topic modelling requires text cleaning, vectorization and 
    latent variable analysis steps. Each of the processing steps will be
    handled by one or more class(es)/function(s), specialized for the 
    task.

    The Modelling pipeline class is responsible for calling the objects 
    required for each processing step in the correct order and passing 
    the correct arguments at each processing step.

    Abstract Properties
    -------------------
    call_order: UniqueList[str].
        step names for each processing step are stored here. Call order 
        specifies the order in which processing steps are taken.

    Abstract Methods
    ----------------
    _get_factory. Returns: callable.
        Builds the method used to retrieve the component handler 
        corresponding to a specific step name.

    _set_factory. Returns: callable.
        Builds the method used to assign a component handler to a 
        specific step name.

    Methods
    -------
    _del_factory. Returns: callable.
        Builds the method used to remove a specific step name from 
        call_order and delete the attribute storing the step name's 
        corresponding component handler.

    _get_step_name. Returns: str.
        Extracts step name from the component argument passed.

    _add_component_property. Returns: None.
        Builds a property to contain a component handler corresponding 
        to a spectific step name in call_order.
    """
    @abstractmethod
    def _get_factory(self, private_name: str) -> callable:
        pass

    @abstractmethod
    def _set_factory(
        self, private_name: str, initial_value: _PipelineComponent
    ) -> callable:
        pass

    def _del_factory(
        self,
        public_name: _step_name,
        private_name: str
    ) -> callable:
        def deletter(self = self) -> None:
            self.call_order.remove(public_name)
            self.__delattr__(private_name)

        return deletter


    def _get_step_name(self, component: _PipelineComponent) -> _step_name:
        """
        Extract step name from the component argument.

        Step name should either be the value returned by 'step_name' key
        in a Model Specification dictionary, or the self_name attribute 
        of a ComponentHandler class, depending on what was passed to 
        component.
        """
        if isinstance(component, dict):
            return component['step_name']
        else:
            component: AbstractComponentHandler
            return component.step_name


    def _add_component_property(
        self, public_name: _step_name, initial_value: _PipelineComponent
    ) -> None:
        """
        Builds a property to store the given pipeline component.

        Parameters
        ``````````

        initial_value: _PipelineComponent.
            An unitialized model; or a model specification dictionary; 
            or an initialized ComponentHandler class. The property's 
            return value will be determined by this parameter.
        """
        private_name = '__' + public_name
        setattr(
            self.__class__,
            public_name,
            property(
                fget = self._get_factory(
                    private_name = private_name
                ),
                fset = self._set_factory(
                    private_name = private_name,
                    initial_value = initial_value
                ),
                fdel = self._del_factory(
                    public_name = public_name,
                    private_name = private_name
                )
            )
        )

    @property
    @abstractmethod
    def call_order(self) -> UniqueList[_step_name]:
        pass