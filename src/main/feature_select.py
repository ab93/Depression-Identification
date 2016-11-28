import os
import sys
import numpy as np
import pandas as pd

import config
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

def get_feature_df(file_, *files):
    feature_df = pd.read_csv(file_)
    if len(files):
        for f in files[0]:
            print f
            feature_df = pd.concat([feature_df, pd.read_csv(f)], axis=1)
            feature_df = feature_df.T.drop_duplicates().T
    
    train_split_df = pd.read_csv(config.TRAIN_SPLIT_FILE, 
                    usecols=['Participant_ID', 'PHQ_Binary'])
    feature_df = feature_df[feature_df['video'].isin(train_split_df['Participant_ID'])]
    print "features: ", feature_df.shape
    print "train_split: ", train_split_df.shape
    train_split_dict = train_split_df.set_index('Participant_ID').T.to_dict()
    del train_split_df
    labels = np.zeros(feature_df.shape[0])
    for i in xrange(feature_df.shape[0]):
        video_id = feature_df.iat[i,0]
        labels[i] = train_split_dict[video_id]['PHQ_Binary']
    feature_df['label'] = pd.Series(labels, index=feature_df.index)
    try:
        feature_df.drop(['video','question','starttime','endtime'], inplace=True, axis=1)
    except ValueError:
        feature_df.drop(['video','question'], inplace=True, axis=1)
    return feature_df

def performPCA(df):
    pca = PCA(n_components=10)
    #df1 = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    df1 = df.drop(['label'], axis=1)
    X=df1.as_matrix()
    pca.fit(X)
    return pca.components_

def removeLowVariance(df):
    #df1 = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    df1 = df.drop(['label'], axis=1)
    column_names=list(df1.columns.values)
    X=df1.as_matrix()
    sel = VarianceThreshold(0.5)
    sel.fit(X)
    selected_feature_idxs = sel.get_support(indices=True)
    selected_features = [column_names[i] for i in selected_feature_idxs]
    print selected_features
    #final_selection = ['frame', 'timestamp','confidence','success']
    final_selection = []
    final_selection.extend(selected_features)
    final_selection.extend(['label'])
    print "First Level: ",len(final_selection)
    final_df = df[final_selection]
    return final_df

def performL1(df):
    #df1 = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    df1 = df.drop(['label'], axis=1)
    column_names=list(df1.columns.values)
    X = df1.as_matrix()
    y = df['label'].values
    svc = LogisticRegression(C=1., penalty='l1', dual=False).fit(X,y)
    model = SelectFromModel(svc,prefit=True)
    selected_feature_idxs = model.get_support(indices=True)
    selected_features = [column_names[i] for i in selected_feature_idxs]
    #final_selection = ['frame', 'timestamp','confidence','success']
    final_selection = []
    final_selection.extend(selected_features)
    final_selection.extend(['label'])
    final_df = df[final_selection]
    print "Second Level: ",len(final_selection)
    return final_df

def selectBestK(df):
    #df1 = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    df1 = df.drop(['label'], axis=1)
    column_names=list(df1.columns.values)
    X = df1.as_matrix()
    y = df['label'].values
    kbest = SelectKBest(f_classif, k=20)
    kbest.fit(X, y)
    selected_feature_idxs = kbest.get_support(indices=True)
    selected_features = [column_names[i] for i in selected_feature_idxs]
    print selected_features
    #final_selection = ['frame', 'timestamp','confidence','success']
    final_selection = []
    final_selection.extend(selected_features)
    final_selection.extend(['label'])
    final_df = df[final_selection]
    print "Fourth Level: ",len(final_selection)
    return final_df

def performRandomForest(df):
    #df1 = df.drop(['frame', 'timestamp', 'confidence', 'success', 'label'], axis=1)
    df1 = df.drop(['label'], axis=1)
    column_names=list(df1.columns.values)
    X = df1.as_matrix()
    y = df['label'].values
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(X, y)
    #print forest.feature_importances_
    important_features = forest.feature_importances_
    ind = [i for i in range(len(important_features))]
    all_pairs = []
    for a, b in zip(important_features, ind):
        all_pairs.append((a, b))
    sorted_pairs=sorted(all_pairs,key=lambda p:p[0],reverse=True)
    selected_feature_idxs = []
    for a, b in sorted_pairs:
        selected_feature_idxs.append(b)
    selected_feature_idxs = selected_feature_idxs[:50]
    selected_features = [column_names[i] for i in selected_feature_idxs]
    print selected_features
    #final_selection = ['frame', 'timestamp','confidence','success']
    final_selection = []
    final_selection.extend(selected_features)
    final_selection.extend(['label'])
    final_df = df[final_selection]
    print "Third Level: ",len(final_selection)
    return final_df

def main():
    file1 = os.path.join(config.D_ND_DIR,sys.argv[1])
    files = [os.path.join(config.D_ND_DIR,argv) for argv in sys.argv[2:]]
    df = get_feature_df(file1, files)
    print df
    #performPCA(df)
    #df = removeLowVariance(df)
    df = performL1(df)
    df = performRandomForest(df)
    df = selectBestK(df)
    #print df
    fileOP = os.path.join(config.SEL_FEAT,"nondiscriminative_linguistic_selected.csv")
    df.to_csv(fileOP,sep=",",index=False)

if __name__ == '__main__':
    main()