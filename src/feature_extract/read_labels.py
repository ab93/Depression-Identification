import pandas as pd

def get_features(data):
    grouped = data.groupby('video')
    X = []
    y = []
    for video, group in grouped:
        X_person = []
        y_person = []
        for i in range(len(group)):
            X_person.append(group.iloc[i].tolist()[1:-1])
            y_person.append(group.iloc[i]['label'])
        X.append(X_person)
        y.append(y_person)
    return X, y


def features(mode,classifier):
    if mode == "visual":
        file = "data/selected_features/train/"+classifier+"_visual_train.csv"
        file_v = "data/selected_features/validation/"+classifier+"_visual_val.csv"
    elif mode == "acoustic":
        file = "data/selected_features/train/"+classifier+"_acoustic_train.csv"
        file_v = "data/selected_features/validation/"+classifier+"_acoustic_val.csv"
    elif mode == "linguistic":
        file = "data/selected_features/train/"+classifier+"_linguistic_train.csv"
        file_v = "data/selected_features/validation/"+classifier+"_linguistic_val.csv"
    print file
    print file_v
    data = pd.read_csv(file)
    data_v = pd.read_csv(file_v)
    return get_features(data),get_features(data_v)
#print features("visual","discriminative")



