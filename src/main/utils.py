import os
import sys
import numpy as np
import pandas as pd
from ..feature_extract.read_labels import features
import config


def oversample(X,y):
    X = np.vstack(X)
    y = np.hstack(y)
    df = pd.DataFrame(X)
    df['labels'] = y
    grouped_df = df.groupby('labels')
    for key, dframe in grouped_df:
        if key == 1:
            sampled_df = dframe
    df = df.append(sampled_df)
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    return X, y


def get_single_mode_data(mode='acoustic', count="all", select="select", category='PN', problem_type='C', feature_scale='False'):
    """
    Get training and validation data for a single mode int
    a particular category

    Args:
        mode: str
        category: str

    Returns:
        X_train: 3D numpy array
        y_train: 2D numpy array
        X_val: 3D numpy array
        y_val: 2D numpy array
    """

    if category == 'PN':
        cat_1 = "positive"
        cat_2 = "negative"
    else:
        cat_1 = "discriminative"
        cat_2 = "nondiscriminative"

    # print mode, category, problem_type, feature_scale

    X_train = [map(np.asarray, features(mode, cat_1, "train", problem_type, feature_scale, count=count)[0]),
               map(np.asarray, features(mode, cat_2, "train", problem_type, feature_scale, count=count)[0])]
    y_train = [map(np.asarray, features(mode, cat_1, "train", problem_type, feature_scale, count=count)[1]),
               map(np.asarray, features(mode, cat_2, "train", problem_type, feature_scale, count=count)[1])]
    X_val = [map(np.asarray, features(mode, cat_1, "val", problem_type, feature_scale)[0]),
             map(np.asarray, features(mode, cat_2, "val", problem_type, feature_scale)[0])]
    y_val = [map(np.asarray, features(mode, cat_1, "val", problem_type, feature_scale)[1]),
             map(np.asarray, features(mode, cat_2, "val", problem_type, feature_scale)[1])]

    X_train = [map(np.asarray, features(mode, cat_1, "train", problem_type, feature_scale, count = count, select = select)[0]),
               map(np.asarray, features(mode, cat_2, "train", problem_type, feature_scale, count = count, select = select)[0])]
    y_train = [map(np.asarray, features(mode, cat_1, "train", problem_type, feature_scale, count = count, select = select)[1]),
               map(np.asarray, features(mode, cat_2, "train", problem_type, feature_scale, count = count, select = select)[1])]
    X_val = [map(np.asarray, features(mode, cat_1, "val", problem_type, feature_scale, select = select)[0]),
             map(np.asarray, features(mode, cat_2, "val", problem_type, feature_scale, select = select)[0])]
    y_val = [map(np.asarray, features(mode, cat_1, "val", problem_type, feature_scale, select = select)[1]),
             map(np.asarray, features(mode, cat_2, "val", problem_type, feature_scale, select = select)[1])]


    return X_train, y_train, X_val, y_val


def get_multi_data(count="all", select="select", category='PN', problem_type='C', feature_scale='regular'):
    X_A_train, y_A_train, X_A_val, y_A_val = get_single_mode_data('acoustic', count, select, category, problem_type, feature_scale)
    X_V_train, y_V_train, X_V_val, y_V_val = get_single_mode_data('visual', count, select, category, problem_type, feature_scale)
    X_L_train, y_L_train, X_L_val, y_L_val = get_single_mode_data('linguistic', count, select, category, problem_type, feature_scale)

    Xs = [X_A_train, X_V_train, X_L_train]
    ys = [y_A_train, y_V_train, y_L_train]
    Xs_val = [X_A_val, X_V_val, X_L_val]
    ys_val = [y_A_val, y_V_val, y_L_val]

    return Xs, ys, Xs_val, ys_val
