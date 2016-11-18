import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

def get_combined_data(file1, file2):
    feature_df = pd.read_csv(file1)
    #print feature_df
    #df = pd.read_csv(file2)
    #print feature_df
    feature_df = feature_df.append(pd.read_csv(file2, header=None))
    
    train_split_df = pd.read_csv(config.TRAIN_SPLIT_FILE, 
                    usecols=['Participant_ID', 'PHQ_Binary'])
    feature_df = feature_df[feature_df['video'].isin(train_split_df['Participant_ID'])]
    print feature_df


def main():
    get_combined_data(os.path.join('../data', sys.argv[1]), 
                    os.path.join('../data', sys.argv[2]))

if __name__ == '__main__':
    main()
