import os
import re
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import config
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC

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
    df = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    X=df.as_matrix()
    pca.fit(X)
    return pca.components_

def removeLowVariance(df):
    df = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    X=df.as_matrix()
    sel = VarianceThreshold(0.95)
    sel.fit(X)
    idxs = sel.get_support(indices=True)
    #print idxs

def performL1(df):
    #vectorizer = CountVectorizer(ngram_range=(1,1), min_df=1)
    df1 = df.drop(['frame', 'timestamp','confidence','success','label'], axis=1)
    X = df1.as_matrix()
    df2 = df[['label']]
    Y = df2.as_matrix()
    print X.shape
    svc = LinearSVC(C=1., penalty='l1', dual=False)
    #svc.fit(X, Y)
    X_train_new = svc.fit_transform(X, Y)
    print X_train_new.shape
    selected_feature_names = svc.coef_
    #selected_feature_names = np.asarray(vectorizer.get_feature_names())[np.flatnonzero(svc.coef_)]
    print selected_feature_names

def analyze_features(df):
    pass

def main():
    file1 = os.path.join(config.D_ND_DIR,sys.argv[1])
    files = [os.path.join(config.D_ND_DIR,argv) for argv in sys.argv[2:]]
    df = get_feature_df(file1, files)
    performPCA(df)
    removeLowVariance(df)
    performL1(df)

if __name__ == '__main__':
    main()