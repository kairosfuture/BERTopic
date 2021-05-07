from math import log10
from multiprocessing import cpu_count
from typing import List, Union, Any, Generator, Iterable
from uuid import uuid4
import os
# import fasttext
from gensim.models import fasttext
from fasttext import load_model, train_unsupervised
from numpy.linalg import norm
import numpy as np


def generator_of(*args) -> Generator[Any, None, None]:
    """
    Returns a `Generator` which either directly uses or pulls data from its parameters.
    :param args: Either non-`Iterable` objects or `Iterable`s. `str` and `bytes` objects are
    considered non-`Iterable` here (even though they are `Iterable`).
    :return: Flattened `Generator` of the data in `args`.
    """
    for arg in args:
        if isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
            yield from arg
        else:
            yield arg


def join_lazy(str_iterable: Iterable[str],
              delimiter: str = '',
              prepend: str = '',
              append: str = '') -> Generator[str, None, None]:
    """
    Makes a `Generator` which puts an extra `delimiter` between `str_iterable`s strings,
    starting with `prepend` and ending with `append` strings.
    This goal here is to make a `Generator` which can act like
        `prepend + delimiter.join(str_iterable) + append`
    but without having to load `str_iterable` in memory.
    This helps with file IO where we don't necessarily need to join in
    memory before writing.
    :param str_iterable: `Iterable` of strings.
    :param delimiter: the strings to put in between `str_iterable`s strings.
    :param prepend: the string that comes before the all the given strings.
    :param append: the string that comes after the all the given strings.
    :return: string `Generator`.
    """
    if isinstance(str_iterable, (str, bytes)):
        raise TypeError("Expected Iterable string, received string or bytes.")
    # iterable to iterator
    str_iterator = iter(str_iterable)
    try:
        first = next(str_iterator)  # consumes first element
        gen_args = [first, map(lambda s: delimiter + s, str_iterator)]
    except StopIteration:
        gen_args = []  # unable to consume first element
    yield from generator_of(prepend, *gen_args, append)


class GensimHelper():
    @classmethod
    def save_docs_to_txt(cls, docs):
        """
        docs: List of list of tokens
        New training mode for *2Vec models (word2vec, doc2vec, fasttext) that allows model training
        to scale linearly with the number of cores (full GIL elimination).
        More: https://github.com/RaRe-Technologies/gensim/blob/3.6.0/CHANGELOG.md
        """
        # save each document to a txt file in which each line represents a single document
        corpus_fname = "{}.txt".format(uuid4().hex)
        with open(corpus_fname, 'w') as corpus_file:
            for string_doc in join_lazy(map(' '.join, docs), delimiter='\n'):
                corpus_file.write(string_doc)
        return corpus_fname

    @classmethod
    def delete_corpus(cls, corpus_path):
        os.remove(corpus_path)


def remove_linebreaks(sentence, replace=''):
    '''
    Remove the linebreaks from the given sentence
    :param sentence: The sentence to manipulate
    :param replace: What will be put instead of linebreaks
    :return: The sentence without any linebreaks
    '''
    return sentence.replace('\r\n', replace).replace('\r', replace).replace('\n', replace)


class FastTextVectorizer():
    _type: str = "FastText"

    def __init__(self, model=None,
                 vector_size=100,
                 max_iters=50,
                 method="skipgram",  # or cbow
                 min_count=2,
                 gensim=False,
                 **kwargs):
        self.model = model
        self.max_iters = max_iters
        self.vector_size = vector_size
        self.method = method
        self.minCount = min_count
        self.isGensim = gensim
        if gensim:
            self.model_name = "/Users/dpc/Desktop/dcipher/Workspace/bertopic_trial/public_model_wos-finetuned_vocab_updated.bin"
        else:
            self.model_name = "wos.bin"

    @classmethod
    def estimate_epoch_count(cls, num_docs):
        # https://towardsdatascience.com/optimising-a-fasttext-model-for-better-accuracy-739ca134100a
        return round(100.0 / (log10(num_docs + 1) + 1e-5))

    def train(self, docs=None, corpus_file=None):
        """
        docs: List of list of tokens
        corpus_file: Path to corpus in format: 1 document per line, words delimited by " "
        """
        if os.path.exists(self.model_name):
            if self.isGensim:
                self.model = fasttext.load_facebook_model(self.model_name)
            else:
                self.model = load_model(self.model_name)
            return
        remove_corpus_after_training = False
        if corpus_file is None and docs is None:
            raise ValueError(
                'At least documents or corpus file path should be provided!')
        if corpus_file is None and docs is not None:
            # get corpus from docs
            corpus_file = GensimHelper.save_docs_to_txt(docs)
            remove_corpus_after_training = True
        if docs is None:
            docs = GensimHelper.read_corpus_into_docs(corpus_file)
        # set number of epochs dynamically
        num_docs = sum([1 for _ in docs])
        estimated_epoch = self.estimate_epoch_count(num_docs)
        epochs = min(estimated_epoch, self.max_iters)
        # train the model
        self.model = train_unsupervised(input=corpus_file,
                                        model=self.method,
                                        dim=self.vector_size,
                                        epoch=epochs,
                                        thread=cpu_count() - 1,
                                        minCount=self.minCount)
        # remove the corpus after training
        if remove_corpus_after_training:
            GensimHelper.delete_corpus(corpus_file)
        self.model.save_model(self.model_name)

    def l2_norm(self, x):
        return norm(x)

    def div_norm(self, x):
        norm_value = self.l2_norm(x)
        if norm_value > 0:
            return x * (1.0 / norm_value)
        else:
            return x

    def gensim_get_sentence_vector(self, string: str):
        token_vectors = [self.div_norm(self.model.wv[token]) for token in string.split()]
        return np.mean(token_vectors, axis=0)

    #     def gensim_get_sentence_vector(self, string: str):
    #         tokens_list = string.split()
    #         resulted_vector = self.div_norm(self.model.wv[tokens_list[0]])
    #         for i in range(1, len(tokens_list)):
    #             resulted_vector += self.div_norm(tokens_list[1])
    #         resulted_vector /= len(tokens_list)
    #         return resulted_vector
    def infer_vector(self, strings: Union[str, List[str]], **kwargs):
        if not isinstance(strings, list):
            raise TypeError("must be list of str")
        if self.isGensim:
            return [self.gensim_get_sentence_vector(remove_linebreaks(string)) for string in
                    strings]
        return [self.model.get_sentence_vector(remove_linebreaks(string)) for string in strings]

    def __call__(self, tokens):
        return self.infer_vector(tokens)

    def encode(self, tokens, show_progress_bar=False):
        if isinstance(tokens, str):
            tokens = [tokens]
        return np.array(self.infer_vector(tokens))


embed = FastTextVectorizer(gensim=True)
embed.train()

import json

path = "/Users/dpc/Desktop/dcipher/Workspace/data/wos-tokens.json"
with open(path, "r") as fp:
    data = json.load(fp)
train_data = [[t["value"] for t in d["tokens"]] for d in data]
# texts = [d["Abstract"] for d in data]
token_docs = [" ".join(doc) for doc in train_data]

# print(len(token_docs))
# for i, doc in enumerate(token_docs):
#     if doc == '':
#         del token_docs[i]
# print(len(token_docs))

print(len(train_data))
for i, doc in enumerate(train_data):
    if len(doc) == 0:
        del train_data[i]
print(len(train_data))

# embeddings = embed(token_docs)

import numpy as np

embeddings = np.loadtxt('/Users/dpc/Desktop/dcipher/Workspace/data/wos_embeddings.txt')

from bertopic import BERTopic

topic_number = 10

model = BERTopic(verbose=True, n_neighbors=15, allow_st_model=True,
                 clustering_method='gmm', nr_topics=topic_number,
                 top_n_words=30)

topics, probs = model.fit_transform(train_data, np.array(embeddings))

init_keywords = model.get_topics()

mmred_keywords = model.mmr_keywords(embedding_model=embed, keywords=init_keywords,
                                    keyword_diversity=0.15)

print(mmred_keywords)
