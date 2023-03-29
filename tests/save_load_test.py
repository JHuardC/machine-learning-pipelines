"""
Testing save-load features to be included in ComponentHandler class
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


### Constants
DATA_PATH: Final[pl.Path] = pl.Path('./petitions_sample.pqt')


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

    df = read_parquet(DATA_PATH)
    # df = df['full_text'].astype('string').str.lower()
    # df = df.apply(preprocess)
