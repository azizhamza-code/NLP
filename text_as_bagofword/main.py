import sys
sys.path.append("..")

from NLP.setup_resources import data_bag_of_words
import pandas as pd
import os 
from pathlib import Path


# setup data if needed
data_bag_of_words()


path_file= Path(__file__).parent.parent
data_path = os.path.join(path_file,"data")

train_data = os.path.join(data_path , "train.tsv")
val_data = os.path.join(data_path , "validation.tsv")
test_data = os.path.join(data_path , "test.tsv")



train = pd.read_csv(train_data,sep='\t')
test = pd.read_csv(test_data,sep='\t')
val = pd.read_csv(val_data,sep='\t')


print(val.columns)
