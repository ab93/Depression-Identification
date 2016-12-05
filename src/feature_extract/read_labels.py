import pandas as pd
from sklearn import preprocessing
import pprint

def get_features(data,classifier_type="C"):
    if classifier_type=="C":
        y_label = 'label'
    else:
        y_label = 'score'
    grouped = data.groupby('video')
    X = []
    y = []
    for video, group in grouped:
        X_person = []
        y_person = []
        for i in range(len(group)):
            X_person.append(group.iloc[i].tolist()[1:-2])
            y_person.append(group.iloc[i][y_label])
        X.append(X_person)
        y.append(y_person)
    return X,y

def features(mode,category,split,problem_type='C',normalize="regular"):
    #print mode,category,split,problem_type,normalize
    if problem_type == "C":
        directory = "classify"
    else:
        directory = "estimate"
    if mode == "visual":
        file = "data/selected_features/"+normalize+"/"+directory+"/"+split+"/"+category+"_visual_"+split+".csv"
    elif mode == "acoustic":
        file = "data/selected_features/"+normalize+"/"+directory+"/"+split+"/"+category+"_acoustic_"+split+".csv"
    elif mode == "linguistic":
        file = "data/selected_features/"+normalize+"/"+directory+"/"+split+"/"+category+"_linguistic_"+split+".csv"
    data = pd.read_csv(file)
    return get_features(data,problem_type)
