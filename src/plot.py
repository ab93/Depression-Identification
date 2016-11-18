import os
import re
import sys
import csv
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import config

def get_combined_data(file1, file2):
    feature_df = pd.read_csv(file1)
    feature_df = feature_df.append(pd.read_csv(file2))
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
    

def calculate_anova(df, filename):
    filename += '.csv'
    columns = df.columns[:-1]
    grouped_df = df.groupby(by='label')
    with open(os.path.join('../results/',filename), 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(["Feature","P-value","F-value"])
        #outfile.write("Feature" + "\t\t" + "P-value" + "\t\t" + "F-value" + "\n")
        for feature in columns:
            data = []
            for key, item in grouped_df:
                temp_df = grouped_df.get_group(key)
                data.append(temp_df.loc[:,feature].values)
            
            f_val, p_val = stats.f_oneway(data[0], data[1])
            #outfile.write(feature + "\t\t" + str(p_val) + "\t\t" + str(f_val) + "\n")
            csv_writer.writerow([feature, p_val, f_val])
    

def main():
    filename = sys.argv[3]
    features_df = get_combined_data(os.path.join('../data', sys.argv[1]), 
                    os.path.join('../data', sys.argv[2]))
    calculate_anova(features_df, filename)

if __name__ == '__main__':
    main()
