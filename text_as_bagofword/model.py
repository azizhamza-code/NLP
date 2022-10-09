from collections import Counter
from typing import Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import json



class model(object):

    def __init__(self):
        self.num_tags = None
        self.y_pred = None
        self.pred_shape = None
        self.class_to_index = None

    def fit(self, x_train, y_train):
        pass

    def transform(self, x_val):
        pass

    def fit_transform(self, x_train, y_val):
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

    def __init__(self,class_to_index: Dict):
        self.class_to_index = class_to_index
        super().__init__()

    def get_one_pred(self):


    def fit(self , x_train , y_train):



def metric(y_test, y_pred):
    metrics = precision_recall_fscore_support(
        y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0],
                   "recall": metrics[1], "f1": metrics[2]}
    print(json.dumps(performance, indent=2))
