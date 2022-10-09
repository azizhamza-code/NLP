from collections import Counter
from typing import Dict, get_origin
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


class model(object):

    def __init__(self):
        self.num_tags = None
        self.y_pred = None
        self.pred_shape = None
        self.class_to_index = None

    def fit(self, x_train, y_train):
        print("is not implemented for this model")
        pass

    def transform(self, x_val):
        print("is not implemented for this model")
        pass

    def fit_transform(self, x_train, y_val):
        print("is not implemented for this model")
        pass


class random_model(model):

    def __init__(self):
        super().__init__()

    def fit(self, x_train, y_train):
        self.y_pred = np.mean(y_train, axis=0)
        self.num_tags = self.y_pred.shape[0]
    # TODO: logging this instead of print

    def get_one_pred(self):
        pred = np.zeros(self.num_tags)
        for i in range(self.num_tags):
            pred[i] = np.random.choice(
                np.arange(2), p=[1-self.y_pred[i], self.y_pred[i]])
        return pred

    def transform(self, x_val):

        if self.y_pred is None:
            print("you need to call fit before claaing transform")
            return None
        else:
            y_pred_val = np.zeros((x_val.shape[0], self.num_tags))
            for i in range(x_val.shape[0]):
                y_pred_val[i, :] = self.get_one_pred()
            return y_pred_val

    def fit_transform(self, x_train, y_train):

        self.fit(x_train, y_train)
        pred = self.transform(y_train)
        return pred


class rule_based(model):

    def __init__(self, class_to_index: Dict):
        super().__init__()
        self.class_to_index = class_to_index

    def get_one_pred(self, row):
        pred = np.zeros(len(self.class_to_index.keys()))
        for token in row:
            if token in self.class_to_index.keys():
                pred[self.class_to_index[token]] = 1
        return pred

    def transform(self, x_train):

        pred = np.zeros(shape=(len(x_train), len(self.class_to_index.keys())))
        for i in range(len(x_train)):
            pred[i] = self.get_one_pred(x_train[i])

        return pred


def metric(y_test, y_pred):
    metrics = precision_recall_fscore_support(
        y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0],
                   "recall": metrics[1], "f1": metrics[2]}
    print(json.dumps(performance, indent=2))

def metric1(y_val, predicted):
    
    print(accuracy_score(y_val,predicted))
    print(f1_score(y_val,predicted,average = 'weighted'))
    print(average_precision_score(y_val,predicted))
