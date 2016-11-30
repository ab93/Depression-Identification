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

def discriminative_feature(mode):
    if mode == "visual":
        file = "data/selected_features/discriminative_visual_selected.csv"
    elif mode == "acoustic":
        file = "data/selected_features/discriminative_acoustic_selected.csv"
    elif mode == "linguistic":
        file = "data/selected_features/discriminative_linguistic_selected.csv"
    # else:
    #     file = "data/selected_features/dummy.csv"
    data = pd.read_csv(file)
    return get_features(data)


def non_discriminative_feature(mode):
    if mode == "visual":
        file = "data/selected_features/nondiscriminative_visual_selected.csv"
    elif mode == "acoustic":
        file = "data/selected_features/nondiscriminative_acoustic_selected.csv"
    elif mode == "linguistic":
        file = "data/selected_features/nondiscriminative_linguistic_selected.csv"
    # else:
    #     file = "data/selected_features/dummy.csv"
    data = pd.read_csv(file)
    return get_features(data)

#print non_discriminative_feature("dummy")

