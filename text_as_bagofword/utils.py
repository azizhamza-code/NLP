
from numbers import Rational
from pathlib import Path
from setup_resources import data_bag_of_words
import pandas as pd
import sys
import os
from prepro import encode_label

sys.path.append(os.path.join("..", "NLP"))


def import_data(ration=0.3, encoders_data=True):

    data_path = get_data_path()
    data_bag_of_words(target_dir=data_path)
    # setup data if needed

    train_data = os.path.join(data_path, "train.tsv")
    val_data = os.path.join(data_path, "validation.tsv")
    test_data = os.path.join(data_path, "test.tsv")

    train = pd.read_csv(train_data, sep='\t')
    test = pd.read_csv(test_data, sep='\t')
    val = pd.read_csv(val_data, sep='\t')

    if ration is not None:

        sample_train = int(len(train)*ration)
        sample_val = int(len(val)*ration)
        sample_test = int(len(test)*ration)

        train = get_squezed(train, sample_train)
        val = get_squezed(val, sample_val)
        test = get_squezed(test, sample_test)

    if encoders_data is True:
        encoders = encode_label()
        encoders_data_path = os.path.join(data_path, "encoding.json")
        encoders = encoders.load(encoders_data_path)
        return train, val, test, encoders

    else:
        return train, val, test

#TODO: add loging


def get_squezed(df, num_sample):
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    df = df[: num_sample]
    return df


def get_data_path():
    path_file = Path(__file__).parent.parent
    data_path = os.path.join(path_file, "data")
    return data_path
