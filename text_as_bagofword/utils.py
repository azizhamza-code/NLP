
import sys
import os
sys.path.append(os.path.join("..","NLP"))

import pandas as pd
from setup_resources import data_bag_of_words

from pathlib import Path


def import_data():

    
    data_path = get_data_path()
    # setup data if needed

    train_data = os.path.join(data_path, "train.tsv")
    val_data = os.path.join(data_path, "validation.tsv")
    test_data = os.path.join(data_path, "test.tsv")

    train = pd.read_csv(train_data, sep='\t')
    data_bag_of_words(target_dir=data_path)
    test = pd.read_csv(test_data, sep='\t')
    val = pd.read_csv(val_data, sep='\t')

    return train, val, test

def get_data_path():
    path_file = Path(__file__).parent.parent
    data_path = os.path.join(path_file, "data")
    return data_path

