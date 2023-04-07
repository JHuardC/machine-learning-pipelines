"""
Simple import test. Checking that the package was installed through pip
as expected.
"""
### Imports
import logging
from tutils import DATA_PATH
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
from mlplines import ModellingPipeline, ComponentHandler


### Set up logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
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

    ### Load and Prep testing data
    corpora = read_parquet(DATA_PATH)
    corpora = corpora['full_text'].str.lower()
    corpora = corpora.apply(preprocess).tolist()

    corpus_dict = Dictionary(corpora)
    corpora = [corpus_dict.doc2bow(text) for text in corpora]

    ### Set-up Pipeline and ComponentHandlers
    sequence = [
        dict(
            step_name = 'tfidf',
            model = TfidfModel,
            init_kwargs = dict(
                smartirs = 'ltc',
                id2word = corpus_dict
            )
        ),
        dict(
            step_name = 'latent_var_model',
            model = LdaModel,
            init_kwargs = dict(
                id2word = corpus_dict,
                num_topics = 30,
                chunksize = 512,
                minimum_probability = 0,
                update_every = 1,
                passes = 5
            )
        )
    ]

    pipeline = ModellingPipeline(sequence = sequence)

    ### Train models
    output = pipeline.train_apply_pipeline(X = corpora)