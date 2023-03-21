# -*- coding: utf-8 -*-
"""
Implement Handler Template CLasses.

Inherit one of these classes and override it's abstract methods to 
create a bespoke implement handler for a given model.

Created on: Tue 03 Jan 2023

@author: Joe Huard
"""
###################
### Imports
from typing import TypeVar, Union
from collections import namedtuple
from topic_modelling_pipelines.pipelines.abc import\
    AbstractModellingPipeline,\
    BaseHandler,\
    ImplementHandlerMixin

from topic_modelling_pipelines.pipelines.hyper_factory import\
    AbstractHyperparameterHandler
from abc import abstractmethod
from collections.abc import Iterable


###################
### Type definitions/aliases
_Model = TypeVar('_Model')
_ModellingPipeline = AbstractModellingPipeline

data_model = namedtuple(
    typename = 'data_model',
    field_names = ['data', 'model'],
    defaults = [None, None],
    module = 'topic_modelling_pipelines.pipelines'
)


###################
### Implement Handlers
class AbstractImplementHandler(BaseHandler, ImplementHandlerMixin):
    """
    Template for Implement Handlers used by ComponentHandler to train 
    and apply a model.

    Root class for all concrete implement handler classes: The progeny 
    of this class will be used by ComponentHandler train and apply 
    models.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    -------------------
    pipeline_env_parameters: tuple[str, ...].
        Property. Contains the models's attribute names whose values 
        are to shared with the modelling pipeline's env_vars dictionary.
        
    Abstract Methods
    ----------------
    _train_with_pipeline_access: data_model.
        Pipeline access means the model is to be treated as 
        untrained; with the data passed treated as if it's the initial 
        data the model will be trained with. 
        
        The pipeline access distinction is necessary, because some 
        models will allow further training on  data - for example, a 
        model could initially be trained on data X using gradient 
        descent for 100 iterations, however a local minima may not have 
        been reached in 100 iterations, so another training cycle may be 
        made using X with a follow up training call.

        Returns a named tuple: data_model. The named tuple contains two
        parameters, data and model. Certain models return the processed 
        training data upon being trained, in such instances this can be 
        passed to the data attribute, otherwise the data attribute 
        should default to None.

    _train_without_pipeline_access: data_model.
        No pipeline access implies the model is to be updated if 
        applicable. Note, the updating the model means further training 
        an already trained model, not updating the hyperparameters.
        
        The pipeline access distinction is necessary, because some 
        models will allow further training on  data - for example, a 
        model could initially be trained on data X using gradient 
        descent for 100 iterations, however a local minima may not have 
        been reached in 100 iterations, so another training cycle may be 
        made using X with a follow up training call.

        Returns a named tuple: data_model. The named tuple contains two
        parameters, data and model. Certain models return the processed 
        training data upon being trained, in such instances this can be 
        passed to the data attribute, otherwise the data attribute 
        should default to None.

    _apply: Iterable.
        Applies model to data.
        
    Methods
    -------
    train. Returns: _Model.
        Trains the model on data X.

    apply. Returns: Iterable.
        Processes data X on trained model.

    train_apply. Returns: tuple[_Model, Iterable].
        Trains and processes data X.

    _train. Returns: namedtuple data_model.
        Contains the logic for when to call _train_with_pipeline_access 
        or _train_without_pipeline_access. Returns a namedtuple 
        data_model, the namedtuple is used because some models will 
        process or 'apply' themselves to the data when training, in such
        cases it is efficient to return both the trained model and 
        processed data.

    _update_pipeline_env. Returns: None.
        Additional arguments derived for the current model and required 
        by future pipeline steps are sent to the pipeline_env dictionary
        using this method.
    """
    step_name: str # instance variable from _BaseHandler

    @property
    @abstractmethod
    def pipeline_env_parameters(self) -> tuple[str, ...]:
        pass
    

    @abstractmethod
    def _train_with_pipeline_access(
        self,
        pipeline: _ModellingPipeline,
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        pass
    

    @abstractmethod
    def _train_without_pipeline_access(
        self, model: _Model, X: Iterable, y: Union[Iterable, None] = None
    ) -> data_model:
        pass
    

    @abstractmethod
    def _apply(self, model: _Model, X: Iterable) -> Iterable:
        pass


    def _update_pipeline_env(
        self, pipeline: _ModellingPipeline, model: _Model
    ) -> None:
        for parameter in self.pipeline_env_parameters:
            if hasattr(model, parameter):
                pipeline.env_vars[parameter] = getattr(model, parameter)
            else:
                continue

    
    def _train(
        self,
        pipeline: Union[_ModellingPipeline, None],
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        if pipeline != None:
            output = self._train_with_pipeline_access(
                pipeline = pipeline, model = model, X = X, y = y
            )

            self._update_pipeline_env(
                pipeline = pipeline, model = output.model
            )

        else:
            output = self._train_without_pipeline_access(
                model = model, X = X, y = y
            )

        return output


    def train(
        self,
        pipeline: Union[_ModellingPipeline, None],
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> _Model:
        _, model = self._train(
            pipeline = pipeline, model = model, X = X, y = y
        )
        return model
    
    
    def apply(
        self,
        pipeline: Union[_ModellingPipeline, None],
        model: _Model,
        X: Iterable
    ) -> Iterable:

        output = self._apply(model = model, X = X)

        if pipeline != None:
            self._update_pipeline_env(
                pipeline = pipeline, model = model
            )

        return output


    def train_apply(
        self,
        pipeline: Union[_ModellingPipeline, None],
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> tuple[Iterable, _Model]:
        output, model = self._train(
            pipeline = pipeline, model = model, X = X, y = y
        )

        if output == None:
            output = self._apply(model = model, X = X)

        return output, model


class SupervisedTrainOnInitImplementer(AbstractImplementHandler):
    """
    Template for Implement Handlers to train and apply supervised 
    learning models that train on itialization.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    -------------------
    pipeline_env_parameters: tuple[str, ...].
        Property. Contains the models's attribute names whose values 
        are to shared with the modelling pipeline's env_vars dictionary.

    data_kwarg: str
        Property. Contains the name of the model's init parameter used 
        to pass training data.

    target_kwarg: str
        Property. Contains the name of the model's init parameter used 
        to pass training data's target variable.
        
    Abstract Methods
    ----------------
    _train_without_pipeline_access: data_model.
        No pipeline access implies the model is to be updated if 
        applicable. Note, updating the model means further training a 
        model, not updating the model's hyperparameters.
        
        The pipeline access distinction is necessary, because some 
        models will allow further training on  data - for example, a 
        model could initially be trained on data X using gradient 
        descent for 100 iterations, however a local minima may not have 
        been reached in 100 iterations, so another training cycle may be 
        made using X with a follow up training call.

        Returns a named tuple: data_model. The named tuple contains two
        parameters, data and model. Certain models return the processed 
        training data upon being trained, in such instances this can be 
        passed to the data attribute, otherwise the data attribute 
        should default to None.

    _apply: Iterable.
        Applies model to data.
        
    Methods
    -------
    train. Returns: _Model.
        Trains the model on data X.

    apply. Returns: Iterable.
        Processes data X on trained model.

    train_apply. Returns: tuple[_Model, Iterable].
        Trains and processes data X.

    _train. Returns: namedtuple data_model.
        Contains the logic for when to call _train_with_pipeline_access 
        or _train_without_pipeline_access. Returns a namedtuple 
        data_model, the namedtuple is used because some models will 
        process or 'apply' themselves to the data when training, in such
        cases it is efficient to return both the trained model and 
        processed data.

    _update_pipeline_env. Returns: None.
        Additional arguments derived for the current model and required 
        by future pipeline steps are sent to the pipeline_env dictionary
        using this method.
    """
    @property
    @abstractmethod
    def data_kwarg(self) -> str:
        pass

    @property
    @abstractmethod
    def target_kwarg(self) -> str:
        pass

    def _train_with_pipeline_access(
        self,
        pipeline: _ModellingPipeline,
        model: _Model,
        X: Iterable,
        y: Iterable
    ) -> data_model:
        hyperparameter_handler: AbstractHyperparameterHandler = getattr(
            getattr(
                pipeline,
                self.step_name
            ), 
            'hyperparameter_handler'
        )

        if not hasattr(hyperparameter_handler, 'initial_kwargs'):
            raise BaseException(
                'train called before model initialization'
            )
        
        elif not isinstance(model, type):
            # triggered when a model is called to be trained multiple
            # times whilst it's model index is greater than the 
            # pipeline's trained threshold
            model = model.__class__

        model = hyperparameter_handler(
            model = model,
            pipeline_external_kwargs = {
                self.data_kwarg: X, self.target_kwarg: y
            },
            pipeline_env_kwargs = pipeline.env_vars
        )

        return data_model(model = model)


class UnsupervisedTrainOnInitImplementer(AbstractImplementHandler):
    """
    Template for Implement Handlers to train and apply unsupervised 
    learning models that train on itialization.

    Required Class Attributes
    -------------------------
    __check_model: Concrete subclass of AbstractCheckModel.
        Class Attribute. Contains the methods for checking if this class
        is the correct handler class for the passed model.

    Abstract Properties
    -------------------
    pipeline_env_parameters: tuple[str, ...].
        Property. Contains the models's attribute names whose values 
        are to shared with the modelling pipeline's env_vars dictionary.

    data_kwarg: str
        Property. Contains the name of the model's init parameter used 
        to pass training data.
        
    Abstract Methods
    ----------------
    _train_without_pipeline_access: data_model.
        No pipeline access implies the model is to be updated if 
        applicable. Note, updating the model means further training a 
        model, not updating the model's hyperparameters.
        
        The pipeline access distinction is necessary, because some 
        models will allow further training on  data - for example, a 
        model could initially be trained on data X using gradient 
        descent for 100 iterations, however a local minima may not have 
        been reached in 100 iterations, so another training cycle may be 
        made using X with a follow up training call.

        Returns a named tuple: data_model. The named tuple contains two
        parameters, data and model. Certain models return the processed 
        training data upon being trained, in such instances this can be 
        passed to the data attribute, otherwise the data attribute 
        should default to None.

    _apply: Iterable.
        Applies model to data.
        
    Methods
    -------
    train. Returns: _Model.
        Trains the model on data X.

    apply. Returns: Iterable.
        Processes data X on trained model.

    train_apply. Returns: tuple[_Model, Iterable].
        Trains and processes data X.

    _train. Returns: namedtuple data_model.
        Contains the logic for when to call _train_with_pipeline_access 
        or _train_without_pipeline_access. Returns a namedtuple 
        data_model, the namedtuple is used because some models will 
        process or 'apply' themselves to the data when training, in such
        cases it is efficient to return both the trained model and 
        processed data.

    _update_pipeline_env. Returns: None.
        Additional arguments derived for the current model and required 
        by future pipeline steps are sent to the pipeline_env dictionary
        using this method.
    """
    @property
    @abstractmethod
    def data_kwarg(self) -> str:
        pass

    def _train_with_pipeline_access(
        self,
        pipeline: _ModellingPipeline,
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        hyperparameter_handler: AbstractHyperparameterHandler = getattr(
            getattr(
                pipeline,
                self.step_name
            ), 
            'hyperparameter_handler'
        )

        if not hasattr(hyperparameter_handler, 'initial_kwargs'):
            raise BaseException(
                'train called before model initialization'
            )
        
        elif not isinstance(model, type):
            # triggered when a model is called to be trained multiple
            # times whilst it's model index is greater than the 
            # pipeline's trained threshold
            model = model.__class__

        model = hyperparameter_handler(
            model = model,
            pipeline_external_kwargs = {self.data_kwarg: X},
            pipeline_env_kwargs = pipeline.env_vars
        )

        return data_model(model = model)