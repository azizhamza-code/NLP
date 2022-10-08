import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np

nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
stemmer = PorterStemmer()


def clean_text(text, lower=True, stem=False, stopwords=STOPWORDS):
    """Clean raw text."""
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub('', text)

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    # Spacing and filters
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)

   # add spacing between objects to be filtered
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower)
                        for word in text.split(" ")])

    return text


class encode_label(object):

    def __init__(self, class_to_index={}):

        self.class_to_index = class_to_index or {}
        self.index_to_class = {k: v for v, k in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):

        classes = np.unique(y.explode().values)

        for i, class_ in enumerate(classes):
            self.class_to_index[i] = class_
        self.class_to_index = {k: v for v, k in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        # returning refernces to so we can using method cascading
        return self
