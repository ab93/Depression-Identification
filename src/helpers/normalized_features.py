import pprint
import pandas as pd
import os
from sklearn import preprocessing
from ..main import config

def get_normalized_features(filename):
    filename_train = filename
    data_train = pd.read_csv(filename_train)
    filename_val = filename.replace("train","val")
    data_val = pd.read_csv(filename_val)
    columns = data_train.columns[1:]
    column = columns[:-2]

    X_train = data_train.as_matrix(columns=column)
    X_val = data_val.as_matrix(columns = column)

    normalizer = preprocessing.Normalizer().fit(X_train)

    transformed_train = normalizer.transform(X_train)
    transformed_val = normalizer.transform(X_val)

    data_normalized_train = pd.DataFrame(transformed_train,columns=column)
    data_normalized_val = pd.DataFrame(transformed_val, columns=column)

    data_normalized_train[['video','label','score']] = data_train[['video','label','score']]
    data_normalized_val[['video','label','score']] = data_val[['video','label','score']]

    write_path_file_train = filename_train.replace("regular","normalize")
    write_path_file_val = filename_val.replace("regular","normalize")

    print filename_train
    print write_path_file_train
    print filename_val
    print write_path_file_val

    data_normalized_train.to_csv(write_path_file_train,index=None)
    data_normalized_val.to_csv(write_path_file_val,index=None)

def normalize_features():
    list_train = [os.path.join(config.SEL_FEAT_TRAIN_REGULAR, fn) for fn in next(os.walk(config.SEL_FEAT_TRAIN_REGULAR))[2]]
    print list_train
    for i in range(len(list_train)):
        get_normalized_features(list_train[i])

normalize_features()
