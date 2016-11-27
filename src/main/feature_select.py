import os
import re
import sys
import csv
import numpy as np
import pandas as pd
import config

def get_feature_df(file_):
    feature_df = pd.read_csv(file_)
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

def analyze_features(df):
    pass

def main():
    print get_feature_df(os.path.join(config.D_ND_DIR,sys.argv[1]))

if __name__ == '__main__':
    main()