"""
Testing save-load features to be included in ComponentHandler class
"""
### Imports
from typing import Final, Union
import pathlib as pl
import pickle as pkl
from functools import partial
from pandas import DataFrame
from gensim.parsing.preprocessing import\
    preprocess_string,\
    strip_tags,\
    strip_short,\
    strip_punctuation,\
    strip_multiple_whitespaces,\
    stem_text,\
    remove_stopwords


### Constants
DATA_PATH: Final[pl.Path] = pl.Path('./test/petitions_sample.pkl')


### functions
def stream_pickle(path: Union[str, pl.Path]):
    with open(path, 'rb') as f:
        while f.peek(1):
            yield pkl.load(f)


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

    df = DataFrame(stream_pickle(DATA_PATH))
    df['full_text'] = df['full_text'].astype('string')
    df['full_text'] = df['full_text'].str.lower()

    text_data = df['full_text'].apply(preprocess)


