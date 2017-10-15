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
            raise FileNotFoundError('File "{}" was not found!' \
                                    .format(filepath))
        self._file = filepath

    @functools.lru_cache(maxsize=1)
    def process(self, n_most_common=1000):
        try:
            line = open(self._file, 'r')
            tokens = [token for token in nltk.word_tokenize(next(line))
                      if not token in stopwords
                      or not is_numeric(token)]
        except IOError as error:
            raise

        tokenset = set(tokens)
        counter = collections.Counter(tokenset) \
                             .most_common(n_most_common - 1)
        

        

        
        
        
        

