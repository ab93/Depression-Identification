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


def features(mode,classifier,split):
    if mode == "visual":
        file = "data/selected_features/"+split+"/"+classifier+"_visual_"+split+".csv"
    elif mode == "acoustic":
        file = "data/selected_features/"+split+"/"+classifier+"_acoustic_"+split+".csv"
    elif mode == "linguistic":
        file = "data/selected_features/"+split+"/"+classifier+"_linguistic_"+split+".csv"
    print file
    data = pd.read_csv(file)
    return get_features(data)
#print features("visual","discriminative","train")



