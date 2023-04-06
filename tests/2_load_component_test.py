"""
Testing load method in ComponentHandler class
"""
### Imports
from typing import Final
import pathlib as pl
from functools import partial
from pandas import read_parquet
from gensim.parsing.preprocessing import\
    preprocess_string,\
    strip_tags,\
    strip_short,\
    strip_punctuation,\
    strip_multiple_whitespaces,\
    stem_text,\
    remove_stopwords

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel

from mlplines import ComponentHandler, ModellingPipeline
from tutils import go_to_ancestor
import pickle

### Constants
DATA_PATH: Final[pl.Path] = go_to_ancestor(pl.Path()).joinpath(
    'tests',
    'test_data',
    'petitions_sample.pqt'
)

OUTPUT_Path: Final[pl.Path] = go_to_ancestor(pl.Path()).joinpath(
    'tests',
    'test_outputs'
)

### Functions
preprocess = partial(
    preprocess_string,
    filters = [
        strip_multiple_whitespaces,
        strip_tags,
        strip_punctuation,
        stem_text,
        strip_short,
        remove_stopwords
    ]
)


########################################################################
########################################################################

if __name__ == '__main__':

    ### Load Attempts
    tfidf = ComponentHandler.load(
        OUTPUT_Path.joinpath('test_1_tfidf.pkl'),
        pickle.load,
        'rb'
    )

    lda = ComponentHandler.load(
        OUTPUT_Path.joinpath('test_1_lda.pkl'),
        pickle.load,
        'rb'
    )

    ### Set-up Pipeline and ComponentHandlers
    pipeline = ModellingPipeline(sequence = [tfidf, lda])

    ### Load and Prep testing data
    corpora = read_parquet(DATA_PATH)
    corpora = corpora['full_text'].str.lower()
    corpora = corpora.apply(preprocess).tolist()

    corpus_dict = Dictionary(corpora)
    corpora = [corpus_dict.doc2bow(text) for text in corpora]

    ### Train models
    output = pipeline.train_apply_pipeline(X = corpora)
