# -*- coding: utf-8 -*-

""" textprocessor.py

This module uses the NCBI E-utilities to retrieve abstracts
and saves them to a file. Multiple processes can be used to
parallelize web scraping.

Requires Python 3.6 or higher.

Example:
        $ python3 dbquery.py -f "summary.txt" -q "breast cancer BRCA1" -n 4

Attributes:
     DbQuery (class): Invoke the ``run`` method after initialization.
         Some queries can take a while to download due to the large number
         of studies (eg. "cancer"). It is often useful to qualify queries
         to make them less generic (eg. "breast cancer BRCA1").

Todo:
    * Command line parsing.
    * Stream instead of bulk download to save memory usage.
    * Pipe parsing and tokenization to reduce IO time.

"""
import os
import functools
import collections
import nltk

stopwords = set(nltk.corpus.stopwords.words('english'))


def is_numeric(val: str) -> bool:
    """
    Check if :param val is coercible to float
    """
    try:
        float(val)
        return True
    except ValueError:
        return False


class TextProcessor:
    def __init__(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'File "{filepath}" was not found!')
        self._file = filepath
        self._vocabulary = {}
        self._corpus = []
        self.n_tokens = None
        self.out_size = None

    @functools.lru_cache(maxsize=1)
    def process(self, n_most_common):

        try:
            with open(self._file, 'r') as _file:
                tokenlists = (nltk.word_tokenize(line) for line in _file)
                tokens = [token.lower() for tokenlist in tokenlists
                          for token in tokenlist
                          if token not in stopwords
                          and not is_numeric(token)]

        except IOError:
            raise
        except Exception:
            raise

        counter = dict(collections.Counter(tokens)
                       .most_common(n_most_common - 1))
        self.out_size = len(counter) + 1
        self._vocabulary = {token: _id for _id, token
                            in enumerate(counter.keys(), start=1)}
        self._vocabulary['UNK'] = 0
        self._corpus = [self._vocabulary[token] for token in tokens
                if token in counter]
        return self._corpus

    def _clear_cache(self):
        self.process.cache_clear()
