# -*coding: utf-8 -*-
""" 
Pipeline class and general component handing classes.

Created on: Thu 01 Sep 2022

@author: Joe Huard
"""
### Imports
import logging
from typing import Callable, Literal, TypeVar, Union
from pathlib import Path
from collections.abc import Iterable, Sequence
from mlplines._component_handler import ComponentHandler
from mlplines.abc import AbstractModellingPipeline
from mlplines._special_entities import UniqueList
from functools import singledispatchmethod

######################
### Logging
logging.getLogger(__name__)

######################
### Define Types
_spec_keys = Literal['ext_lookup', 'step_name', 'model', 'init_kwargs']
_PathLike = TypeVar('_PathLike', str, Path)
_step_name = str
_Model = TypeVar('_Model')
_init_kwargs = dict
ModelSpec = dict[
    _spec_keys,
    Union[_PathLike, _step_name, type[_Model], _init_kwargs]
]

PipelineComponent = Union[ModelSpec, ComponentHandler]
_ModellingPipeline = TypeVar('_ModellingPipeline', bound = 'ModellingPipeline')

######################
### Modelling Pipeline
class ModellingPipeline(AbstractModellingPipeline):
    """
    Coordinates the processing of data that requires squential use of 
    multiple objects/models.

    Attributes
    ----------
    call_order: UniqueList[str].
        Property. Contains the class attribute names for each step in 
        the pipeline, in the order these steps are to be applied to 
        data.

    env_vars: dict.
        Property. A shared space for additional kwargs to be stored for 
        future use by component models in later pipeline processing 
        steps.

    trained_threshold: int.
        Property. Used to monitor how much of the pipeline sequence has 
        been trained. The pipeline segment made of components with 
        call_order index values below this threshold, is designated as 
        the 'trained' segment.

    update_threshold: int.
        Property. The pipeline segment made of components with 
        call_order index values greater than or equal to this threshold,
        is designated as the 'updates required' segment.

    Methods
    -------
    update_component. Returns: None.
        Updates the hyperparameters of a specific model, by referencing 
        their step_name.

    train_component. Returns: None.
        Trains a specific model by referencing their step_name.

    apply_component. Returns: Iterable.
        Applies a specific model to data, the model is retreived by 
        referencing their step_name.

    train_apply_component. Returns: Iterable.
        Trains a specific model and applies the model to the training 
        data, the model is retreived by referencing their step_name.

    train_partial_pipeline. Returns: None.
        Sequentially trains the models that make up a segment of the 
        pipeline.

    apply_partial_pipeline. Returns: dict[str, Iterable].
        Sequentially applies the models that make up a segment of the 
        pipeline to data.

    train_apply_partial_pipeline. Returns: dict[str, Iterable].
        Sequentially trains and applies each model that makes up a 
        segment of the pipeline to the training data.

    train_pipeline. Returns: None.
        Sequentially trains each model in the pipeline.

    apply_pipeline. Returns: dict[str, Iterable].
        Sequentially applies each model in the pipeline.

    train_apply_pipeline. Returns: dict[str, Iterable].
        Sequentially trains and applies each model in the pipeline to 
        the training data.
    """
    def _get_factory(self: _ModellingPipeline, private_name: str) -> callable:
        def getter(self: _ModellingPipeline) -> ComponentHandler:
            return getattr(self, private_name)

        return getter


    def _set_factory(
        self: _ModellingPipeline, 
        private_name: str, 
        initial_value: PipelineComponent
    ) -> callable:
        def setter(self: _ModellingPipeline, value: PipelineComponent) -> None:
            if isinstance(value, ComponentHandler):
                setattr(self, private_name, value)
            else:
                setattr(self, private_name, ComponentHandler(**value))

        setter(self, initial_value)

        return setter


    def _append_component(
        self: _ModellingPipeline,
        initial_value: PipelineComponent
    ) -> None:
        public_name = self._get_step_name(component = initial_value)
        self.call_order.append(public_name)
        self._add_component_property(
            public_name = public_name,
            initial_value = initial_value
        )


    def __init__(
        self,
        sequence: Sequence[PipelineComponent],
        ext_lookup: Union[_PathLike, None] = None
    ):
        """
        Processes data through the steps given in sequence.

        Parameters
        ----------
        sequence: indexed container of PipelineComponents.
            The steps of the pipeline are specified by this parameter. 
            The index of each element determines when the associated 
            model will be called by the ModellingPipeline.

            PipelineComponents are dictionaries containing the 
            following key-value pairs:

                step_name:
                    str, reference name for the step's ComponentHandler.
                    The value will also become a class attribute name,
                    containing the ComponentHandler.

                model:
                    The class that will perform the data processing.

                init_kwargs:
                    dict, kwargs to initialize the model with.

                (Optional) ext_lookup:
                    Pathlike, Models require HyperparmeterHandler and 
                    ImplementHandler classes to be used by 
                    ModellingPipeline, if custom handlers for the given 
                    model are contained in a separate script the 
                    pathlike given here will be used to load the correct
                    handlers. An ext_lookup kwarg given in the sequence
                    dictionary will override the ext_lookup below.

        ext_lookup: Pathlike.
            Optional argument. Path to the folder with scripts
            containing custom  HyperparmeterHandler and ImplementHandler
            classes. All HyperparmeterHandler and ImplementHandler
            classes for a specific package should be grouped into a
            single script named {package-name}_extensions.py
        """
        self.__call_order: UniqueList[_step_name] = UniqueList()
        self.__env_vars = dict()
        self.__trained_threshold = 0
        self.__update_threshold = 0

        for component in sequence:

            temp = component
            self.add_ext_lookup(component, ext_lookup)
            self._append_component(initial_value = temp)


    @singledispatchmethod
    def add_ext_lookup(self, component: object, ext_lookup: _PathLike) -> None:
        """
        Called when ComponentHandler instance is passed in sequence.
        """
        if isinstance(component, ComponentHandler):
            # Handler classes will have already been loaded
            pass
        else:
            raise TypeError(f'{component}, not ComponentHandler type')

    @add_ext_lookup.register(dict)
    def _(self, component: dict, ext_lookup: _PathLike) -> None:

        if ('ext_lookup' not in component) and (ext_lookup != None):
            component['ext_lookup'] = ext_lookup
    

    @property
    def call_order(self) -> UniqueList[_step_name]:
        # Note: no setter function modified using inplace functions only
        return self.__call_order
    
    @property
    def env_vars(self) -> dict:
        # Note: no setter function modified using inplace functions only
        return self.__env_vars

    @property
    def trained_threshold(self) -> int:
        return self.__trained_threshold
    @trained_threshold.setter
    def trained_threshold(self, value: int) -> None:

        if value not in range(0, len(self.call_order) + 1):
            error = ''.join(
                [
                    "Value passed to trained_threshold is out of range:\n",
                    f"\tvalue passed: {value}\n",
                    "\taccepted range (inclusive): (0, ",
                    f"{len(self.call_order)})"
                ]
            )
            raise ValueError(error)

        elif value > self.__update_threshold:
            error = ''.join(
                [
                    "trained_threshold value cannot be larger than ",
                    "update_threshold value:\n",
                    f"\tupdate_threshold value: {self.__update_threshold}\n"
                    f"\tvalue passed to trained_threshold: {value}"
                ]
            )
            raise ValueError(error)

        self.__trained_threshold = value


    @property
    def update_threshold(self) -> int:
        return self.__update_threshold
    @update_threshold.setter
    def update_threshold(self, value: int) -> None:

        if value not in range(len(self.call_order) + 1):
            error = ''.join(
                [
                    "Value passed to trained_threshold is out of range:\n",
                    f"\tvalue passed: {value}\n",
                    "\taccepted range (inclusive): (0, ",
                    f"{len(self.call_order)})"
                ]
            )
            raise IndexError(error)

        self.__update_threshold = value

    
    def _check_name_in_call_order(
        self, component_name: str
    ) -> None:
        """
        Throws error if value passed is not in call_order.
        """
        assert component_name in self.call_order, \
            f'component_name: {component_name}, not a part of this pipeline.'


    def _get_idx_and_handler(
        self, component_name: _step_name
    ) -> tuple[int, ComponentHandler]:
        """
        Retreives call_order index and associated ComponentHandler 
        attribute.
        """
        component: ComponentHandler = getattr(self, component_name)
        return self.call_order.index(component_name), component


    def _update_component(
        self, component_idx: int, component: ComponentHandler, **updates
    ) -> None:
        """
        Updates model's hyperparameters and pipeline's update_threshold.
        """
        component.update_kwargs(
            pipeline_external_kwargs = updates,
            pipeline_env_kwargs = self.env_vars
        )
        
        if component_idx <= self.trained_threshold:
            self.update_threshold = component_idx + 1
            self.trained_threshold = component_idx


    def update_component(
        self, component_name: _step_name, **updates
    ) -> None:
        """
        Updates the hyperparameters of a specific model by referencing 
        their step_name.
        """
        self._check_name_in_call_order(component_name = component_name)
        component_idx, component = self._get_idx_and_handler(component_name)
        self._update_component(component_idx, component, **updates)


    def _train_component(
        self,
        component_idx: int,
        component: ComponentHandler,
        X: Iterable,
        y: Union[Iterable, None] = None,
        **updates
    ) -> None:
        """
        Trains model and updates pipeline's trained_threshold.
        """
        if (component_idx >= self.update_threshold) or len(updates):
            self._update_component(component_idx, component, **updates)

        if component_idx > self.trained_threshold:

            component.train(pipeline = self, X = X, y = y)
        
        elif component_idx == self.trained_threshold:

            component.train(pipeline = self, X = X, y = y)
            self.trained_threshold += 1

        else:
            
            component.train(X = X, y = y)


    def train_component(
        self,
        component_name: _step_name,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> None:
        """
        Trains a specific model by referencing their step_name.

        Parameters
        ----------
        component_name: str.
            Step name, found in call_order. Attribute of this name 
            contains the relevant component handler.

        X: Iterable.
            Data the models will be trained on.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.
        """
        self._check_name_in_call_order(component_name = component_name)
        component_idx, component = self._get_idx_and_handler(component_name)
        
        if component_idx > self.trained_threshold:
            # TODO: log warning the component state will remain as untrained
            pass 

        self._train_component(component_idx, component, X, y = y)


    def _apply_component(
        self,
        component_idx: int,
        component: ComponentHandler,
        X: Iterable,
        y: Union[Iterable, None] = None,
        **updates
    ) -> Iterable:
        return component.apply(X = X)


    def apply_component(
        self, component_name: _step_name, X: Iterable
    ) -> Iterable:
        """
        Applies a specific model to data, the model is retreived by 
        referencing their step_name.

        Parameters
        ----------
        component_name: str.
            Step name, found in call_order. Attribute of this name 
            contains the relevant component handler.

        X: Iterable.
            Data the model will be applied to.
        """
        # TODO: log warnings when component is untrained
        self._check_name_in_call_order(component_name = component_name)
        idx, component = self._get_idx_and_handler(component_name)
        return self._apply_component(idx, component, X)


    def _train_apply_component(
        self,
        component_idx: int,
        component: ComponentHandler,
        X: Iterable,
        y: Union[Iterable, None] = None,
        **updates
    ) -> Iterable:
        if component_idx >= self.update_threshold:
            self._update_component(component_idx, component, **updates)

        if component_idx > self.trained_threshold:
            return component.train_apply(pipeline = self, X = X, y = y)
        
        elif component_idx == self.trained_threshold:
            output = component.train_apply(pipeline = self, X = X, y = y)
            self.trained_threshold += 1
            return output

        else:
            return component.train_apply(X = X, y = y)


    def train_apply_component(
        self,
        component_name: _step_name,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> Iterable:
        """
        Trains a specific model and applies the model to the training 
        data, the model is retreived by referencing their step_name.

        Parameters
        ----------
        component_name: str.
            Step name, found in call_order. Attribute of this name 
            contains the relevant component handler.

        X: Iterable.
            Data the model will be trained on, then applied to.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.
        """
        self._check_name_in_call_order(component_name = component_name)
        idx, component = self._get_idx_and_handler(component_name)
        return self._train_apply_component(idx, component, X, y)


    def _get_partial_pipeline(
        self,
        from_to: Union[
            _step_name, 
            tuple[
                Union[_step_name, int, None], 
                Union[_step_name, int, None]
            ]
        ]
    ) -> list[_step_name]:
        """
        Retreive a segment of the pipeline's call_order attribute.

        Parameter
        ---------
        from_to: str or tuple.
            Declare components at the start and end of the desired 
            pipeline segment. If None or int types are passed in the 
            tuple, they will be used as index values for call_order.
        """
        if isinstance(from_to, str):
            self._check_name_in_call_order(from_to)
            return [from_to]
        else:
            start, end = from_to

            if isinstance(start, str):
                self._check_name_in_call_order(start)
                start = self.call_order.index(start)

            if isinstance(end, str):
                self._check_name_in_call_order(end)
                end = self.call_order.index(end) + 1

            return self.call_order[start: end]


    def _use_component(
        self,
        component_idx: int,
        component: ComponentHandler, 
        updates: dict,
        method: Callable, 
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> Union[Iterable, None]:
        """
        Call an internal method to apply to a specific model.

        Parameters
        ----------
        component_idx: int.
            The position in call_order for the component handler class 
            passed.
        
        component: ComponentHandler class.
            The component handler to be updated and passed to method.

        updates: dict.
            Nested dictionary of hyperparameter updates for each model 
            in the Pipeline segment being called.

        method: callable.
            One of the methods _train, _apply, _train_apply to use on 
            the component give.

        X: Iterable.
            Data the models will be applied to / trained on.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.
        """
        return method(component_idx, component, X, y, **updates)


    def _use_partial_pipeline(
        self,
        loop_method: Callable[
            [int, ComponentHandler, Iterable], Union[Iterable, None]
        ],
        final_method: Callable[
            [int, ComponentHandler, Iterable], Union[Iterable, None]
        ],
        from_to: Union[
            _step_name, 
            tuple[
                Union[_step_name, int, None], 
                Union[_step_name, int, None]
            ]
        ],
        X: Iterable,
        y: Union[Iterable, None] = None, 
        updates: Union[dict[_step_name, dict], None] = None
    ) -> dict[_step_name, Union[Iterable, None]]:
        """
        Call a method to apply sequentially to a segment of the pipeline

        Parameters
        ----------
        loop_method: callable.
            One of the methods _train, _apply, _train_apply to use on 
            all but the final component in the pipeline segment.

        final_method: callable.
            One of the methods _train, _apply, _train_apply to use on 
            the final component in the pipeline segment.

        from_to: str or tuple.
            Declare components at the start and end of the desired 
            pipeline segment. If None or int types are passed in the 
            tuple, they will be used as index values for call_order.

        X: Iterable.
            Data the models will be applied to / trained on.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.

        updates: dict.
            Nested dictionary of hyperparameter updates for each model 
            in the Pipeline segment being called.
        """
        partial_pipeline = self._get_partial_pipeline(from_to = from_to)

        final_component_name  = partial_pipeline.pop(-1)
        final_idx, final_component = self._get_idx_and_handler(
            final_component_name
        )

        next_input = X

        # Set updates as dictionary if None is passed to updates
        if updates == None:
            updates = dict()

        # Initialize outputs dictionary
        outputs = dict()

        if partial_pipeline:

            step = partial_pipeline.pop(0)

            idx, component = self._get_idx_and_handler(step)

            if idx <= self.trained_threshold:
                self.trained_threshold = idx
                self.update_threshold = idx

            outputs[step] = self._use_component(
                component_idx = idx,
                component = component,
                updates = updates.get(step, dict()),
                method = loop_method,
                X = next_input
            )
            
            next_input = outputs.get(step)

            # loop through partial_pipeline
            for step in partial_pipeline:

                idx, component = self._get_idx_and_handler(step)

                outputs[step] = self._use_component(
                    component_idx = idx,
                    component = component,
                    updates = updates.get(step, dict()),
                    method = loop_method,
                    X = next_input
                )
                
                next_input = outputs.get(step)
        
        else:
            
            if final_idx <= self.trained_threshold:
                self.trained_threshold = final_idx
                self.update_threshold = final_idx

        outputs[final_component_name] = self._use_component(
            component_idx = final_idx,
            component = final_component,
            updates = updates.get(final_component_name, dict()),
            method = final_method,
            X = next_input,
            y = y
        )

        return outputs


    def apply_partial_pipeline(
        self, 
        from_to: Union[
            _step_name, 
            tuple[
                Union[_step_name, int, None], 
                Union[_step_name, int, None]
            ]
        ],
        X: Iterable
    ) -> dict[str, Iterable]:
        """
        Partially apply the pipeline to from a specific component_step
        onwards.

        Parameters
        ----------
        from_to: str or tuple.
            Declare components at the start and end of the desired 
            pipeline segment. If None or int types are passed in the 
            tuple, they will be used as index values for call_order.

        X: Iterable.
            Data the models will be applied to.

        Returns
        -------
        Dictionary whose key-value pairs are the model's step name and
        the outputs from applying each model respectively.
        """
        return self._use_partial_pipeline(
            self._apply_component,
            self._apply_component,
            from_to,
            X
        )


    def train_partial_pipeline(
        self, 
        from_to: Union[
            _step_name, 
            tuple[
                Union[_step_name, int, None], 
                Union[_step_name, int, None]
            ]
        ],
        X: Iterable,
        y: Union[Iterable, None] = None, 
        updates: Union[dict[_step_name, dict], None] = None,
    ) -> None:
        """
        Train components of the pipeline from a specific component_step
        onwards.

        Parameters
        ----------
        from_to: str or tuple.
            Declare components at the start and end of the desired 
            pipeline segment. If None or int types are passed in the 
            tuple, they will be used as index values for call_order.

        X: Iterable.
            Data the models will be trained on.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.

        updates: dict.
            Nested dictionary of hyperparameter updates for each model 
            in the Pipeline segment being called.
        """
        self._use_partial_pipeline(
            self._train_apply_component,
            self._train_component,
            from_to,
            X,
            y,
            updates
        )
        return None


    def train_apply_partial_pipeline(
        self, 
        from_to: Union[
            _step_name, 
            tuple[
                Union[_step_name, int, None], 
                Union[_step_name, int, None]
            ]
        ],
        X: Iterable,
        y: Union[Iterable, None] = None, 
        updates: Union[dict[_step_name, dict], None] = None,
    ) -> dict[str, Iterable]:
        """
        Train, then apply components of the pipeline between specific 
        component_steps.

        Parameters
        ----------
        from_to: str or tuple.
            Declare components at the start and end of the desired 
            pipeline segment. If None or int types are passed in the 
            tuple, they will be used as index values for call_order.

        X: Iterable.
            Data the models will be trained on and applied to.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.

        updates: dict.
            Nested dictionary of hyperparameter updates for each model 
            in the Pipeline segment being called.

        Returns
        -------
        Dictionary whose key-value pairs are the model's step name and
        the outputs from applying each model respectively.
        """
        return self._use_partial_pipeline(
            self._train_apply_component,
            self._train_apply_component,
            from_to,
            X,
            y,
            updates
        )
        

    def train_pipeline(
        self,
        X: Iterable,
        y: Union[Iterable, None] = None,
        updates: Union[dict[_step_name, dict], None] = None,
    ) -> None:
        """
        Train each component of the pipeline.

        Parameters
        ----------
        X: Iterable.
            Data the models will be trained on.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.

        updates: dict.
            Nested dictionary of hyperparameter updates for each model 
            in the Pipeline segment being called.
        """
        return self.train_partial_pipeline(
            from_to = (None, None), 
            X = X,
            y = y,
            updates = updates
        )
        

    def apply_pipeline(
        self,
        X: Iterable
    ) -> dict[str, Iterable]:
        """
        Apply each component of the pipeline to data.

        Parameters
        ----------
        X: Iterable.
            Data the models will be applied to.

        Returns
        -------
        Dictionary whose key-value pairs are the model's step name and
        the outputs from applying each model respectively.
        """
        return self.apply_partial_pipeline(
            from_to = (None, None), 
            X = X
        )
        

    def train_apply_pipeline(
        self,
        X: Iterable,
        y: Union[Iterable, None] = None, 
        updates: Union[dict[_step_name, dict], None] = None,
    ) -> dict[str, Iterable]:
        """
        Train and apply each component of the pipeline.

        Parameters
        ----------
        X: Iterable.
            Data the models will be trained on and applied to.

        y: Iterable or None.
            Optional argument, target values for supervised learning 
            models.

        updates: dict.
            Nested dictionary of hyperparameter updates for each model 
            in the Pipeline segment being called.

        Returns
        -------
        Dictionary whose key-value pairs are the model's step name and
        the outputs from applying each model respectively.
        """
        return self.train_apply_partial_pipeline(
            from_to = (None, None), 
            X = X,
            y = y,
            updates = updates
        )