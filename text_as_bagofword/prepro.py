from importlib.resources import contents
import json
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
            self.class_to_index[class_] = i
        self.index_to_class = {k: v for v, k in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        # returning refernces to so we can using method cascading
        return self

    def encode(self, y):
        y_coded = np.zeros(len(self.class_to_index))
        for y_item in y:
            y_coded[self.class_to_index[y_item]] = 1
        return y_coded

    def decode(self, y_encoded):

        y_decoded = [self.index_to_class[index_]
                     for index_ in np.where(y_encoded == 1)[0]]
        return y_decoded

    def save(self,fp):

        with open(fp , "w") as fp:
            contents = {"class_to_inndex" : self.class_to_index}
            json.dump(contents,fp , indent=4 , sort_keys=False)

    @classmethod
    def load(cls , fp):
        with open(fp , 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
