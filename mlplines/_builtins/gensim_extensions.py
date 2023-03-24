# UTF-8 Encoding
"""
Hyperparameter and Implement handlers for gensim models.

Created on: Wed 01 Feb 2023

@Author: Joe Huard
"""
### Imports
from typing import Union, TypeVar
from mlplines.abc import AbstractModellingPipeline
from mlplines.hyper_factory import TrainOnInitializationHandler
from mlplines.implement_factory import\
    UnsupervisedTrainOnInitImplementer, data_model

from mlplines.check_factory import SubclassOfKey
from collections.abc import Iterable
from collections import OrderedDict
from gensim.interfaces import TransformationABC
from gensim.models.basemodel import BaseTopicModel

##########
### Typing
_Model = TypeVar('_Model')
ModellingPipeline = AbstractModellingPipeline
_GensimTransform = TransformationABC
_GensimTopicModel = BaseTopicModel

### Gensim check model and handlers
class SubclassOfGensim(SubclassOfKey):

    @classmethod
    def get_key(
        cls
    ) -> tuple[type[TransformationABC], type[BaseTopicModel]]:
        return TransformationABC, BaseTopicModel


class GensimHyperparameterHandler(TrainOnInitializationHandler):

    __check_model = SubclassOfGensim

    @property
    def data_kwarg(self) -> str:
        return 'corpus'

    @property
    def _relevant_pipeline_env_keys(self) -> set[str]:
        return {'id2word'}


class GensimNotUpdatable(SubclassOfGensim):

    @classmethod
    def get_key(
        cls
    ) -> OrderedDict[
        str, Union[str, tuple[type[TransformationABC], type[BaseTopicModel]]]
    ]:
        return OrderedDict(
            method_modifier = 'not ',
            method = 'update', 
            parents = super().get_key()
        )
    
    @classmethod
    def get_match_description(cls, realization_class: type) -> str:
        method_modifier, method, parents = tuple(cls.get_key().values())
        if not isinstance(parents, tuple):
            parents = (parents,)

        description = [
            f"{realization_class.__name__} is returned if:\n\n",
            "\t1. The model passed is a subclass of one or more of the ",
            "following:\n\n",
            *(f"\t\t- {el.__name__}\n" for el in parents),
            "\n",
            f"And:\n\n\t2. The method '{method}' is {method_modifier}a method",
            " in the specification class"
        ]
        return ''.join(description)
    

    @classmethod
    def check_model(cls, model: _Model) -> bool:
        _, method, parents = tuple(cls.get_key().values())
        return all((issubclass(model, parents), method not in dir(model)))


class ImplementGensimTransformer(UnsupervisedTrainOnInitImplementer):

    __check_model = GensimNotUpdatable

    @property
    def pipeline_env_parameters(self) -> tuple[str, ...]:
        return tuple()

    @property
    def data_kwarg(self) -> str:
        return 'corpus'
    
    def _train_without_pipeline_access(
        self,
        model: _GensimTransform,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        raise NotImplementedError(
            "Gensim Tranform classes do not update attributes with new data"
        )

    def _apply(self, model: _GensimTransform, X: Iterable) -> Iterable:
        return model[X]


class GensimUpdatable(GensimNotUpdatable):

    @classmethod
    def get_key(
        cls
    ) -> OrderedDict[
        str, Union[str, tuple[type[TransformationABC], type[BaseTopicModel]]]
    ]:
        alias_key = super().get_key()
        alias_key['method_modifier'] = ''
        
        return alias_key

    @classmethod
    def check_model(cls, model: _Model) -> bool:
        _, method, parents = tuple(cls.get_key().values())
        return all((issubclass(model, parents), method in dir(model)))


class ImplementGensimTopic(UnsupervisedTrainOnInitImplementer):
    __check_model = GensimUpdatable

    @property
    def pipeline_env_parameters(self) -> tuple[str, ...]:
        return tuple()

    @property
    def data_kwarg(self) -> str:
        return 'corpus'
    
    def _train_without_pipeline_access(
        self,
        model: _GensimTopicModel,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        model.update(X)
        return data_model(None, model)
            
    
    def _apply(self, model: _GensimTopicModel, X: Iterable) -> Iterable:
        return model[X]