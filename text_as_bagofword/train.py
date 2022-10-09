import numpy as np
from nltk.tokenize import WhitespaceTokenizer
import snoop
import nltk

# TODO: try different tokenizer
#nltk.download('punkt')


class vectorizer_scratch(object):

    def __init__(self, n_most, del_stop_word=True):

        self.n_most = n_most

    @staticmethod
    def get_num_words_corpus(data):
        tk = WhitespaceTokenizer()
        data = data.apply(lambda x: tk.tokenize(x))
        data = data.values
        data = np.hstack(data)
        return np.unique(data)
