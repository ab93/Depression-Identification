import pandas as pd
from sklearn import preprocessing
import pprint
from ..main import config


def get_features(data, split, classifier_type="C"):
    if classifier_type == "C":
        y_label = 'label'
    else:
        y_label = 'score'
    grouped = data.groupby('video')
    X = []
    y = []
    if split != "test":
        for video, group in grouped:
            X_person = []
            y_person = []
            for i in range(len(group)):
                X_person.append(group.iloc[i].tolist()[1:-2])
                y_person.append(group.iloc[i][y_label])
            X.append(X_person)
            y.append(y_person)
        return X, y
    elif split == "test":
        for video, group in grouped:
            X_person = []
            for i in range(len(group)):
                X_person.append(group.iloc[i].tolist()[1:])
            X.append(X_person)
        return X


def features(mode, category, split, problem_type='C', feature_scale=False, count="all", select = "select"):
    normalize = 'normalize' if feature_scale else 'regular'
    if problem_type == "C":
        directory = "classify"
    else:
        directory = "estimate"
    if select == "select":
        sel = "selected_features"
    else:
        sel = "all_features"
    if mode == "visual":
        file_ = "data/"+sel+"/"+normalize+"/"+directory+"/"+split+"/"+category+"_visual_"+split+".csv"
    elif mode == "acoustic":
        file_ = "data/"+sel+"/"+normalize+"/"+directory+"/"+split+"/"+category+"_acoustic_"+split+".csv"
    elif mode == "linguistic":
        file_ = "data/"+sel+"/"+normalize+"/"+directory+"/"+split+"/"+category+"_linguistic_"+split+".csv"
    data = pd.read_csv(file_)
    if split == "train" and count != "all":
        split_file = config.TRAIN_SPLIT_FILE
        split_df = pd.read_csv(split_file,usecols=['Participant_ID'])
        split_df = split_df.loc[:int(count)-1]
        data = data[data['video'].isin(split_df['Participant_ID'])]
    return get_features(data, split, problem_type)
