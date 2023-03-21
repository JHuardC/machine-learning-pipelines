# UTF-8 Encoding
"""
Hyperparameter and Implement handlers for the topic modelling pipelines
package.

Created on: Wed 01 Feb 2023

@Author: Joe Huard
"""
### Imports
from typing import TypeVar, Union
from topic_modelling_pipelines.pipelines.abc import AbstractModellingPipeline
from topic_modelling_pipelines.pipelines.check_factory import SubclassOfKey
from topic_modelling_pipelines.preprocessing.spacy_modifier import\
    SpaCyPipelineModifierABC

from topic_modelling_pipelines.pipelines.hyper_factory import\
    SimpleHyperparameterHandler

from topic_modelling_pipelines.pipelines.implement_factory import\
    AbstractImplementHandler, data_model

from collections.abc import Iterable

### Typing
_Model = TypeVar('_Model')
ModellingPipeline = AbstractModellingPipeline

### Preprocessing check model and handlers
class SubclassOfSpacypreproc(SubclassOfKey):
    
    @classmethod
    def get_key(cls) -> type[SpaCyPipelineModifierABC]:
        return SpaCyPipelineModifierABC


spacy_pipeline = SpaCyPipelineModifierABC

class SpacyHyperParameterHandler(SimpleHyperparameterHandler):

    __check_model = SubclassOfSpacypreproc

    @property
    def _relevant_pipeline_env_keys(self) -> set[str]:
        return set()
        

class SpacyImplementHandler(AbstractImplementHandler):

    __check_model = SubclassOfSpacypreproc

    @property
    def pipeline_env_parameters(self) -> tuple[str, ...]:
        return ('id2word', )

    def _train_with_pipeline_access(
        self,
        pipeline: ModellingPipeline,
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        if hasattr(model, 'id2word'):
            delattr(model, 'id2word')
        
        data = model(X)

        return data_model(data, model)
        
        
    def _train_without_pipeline_access(
        self,
        model: _Model,
        X: Iterable,
        y: Union[Iterable, None] = None
    ) -> data_model:
        data = model(X)
        return data_model(data, model)
    

    def _apply(self, model: spacy_pipeline, X: Iterable) -> Iterable:
        return model(X)