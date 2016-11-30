import os
import numpy as np
import pandas as pd

import config
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

def get_feature_df(train, file_, *files):
    # Set directory based on Train and Validation
    if train == 'y':
        split_file = config.TRAIN_SPLIT_FILE
    else:
        split_file = config.VAL_SPLIT_FILE

    # Append file columns to a single data frame
    feature_df = pd.read_csv(file_)
    if len(files):
        for f in files[0]:
            print f
            feature_df = pd.concat([feature_df, pd.read_csv(f)], axis=1)
            feature_df = feature_df.T.drop_duplicates().T

    # Trim data frame to hole only train/validation records
    split_df = pd.read_csv(split_file,usecols=['Participant_ID', 'PHQ_Binary'])
    feature_df = feature_df[feature_df['video'].isin(split_df['Participant_ID'])]

    # Populate labels accordingly
    split_dict = split_df.set_index('Participant_ID').T.to_dict()
    del split_df
    labels = np.zeros(feature_df.shape[0])
    for i in xrange(feature_df.shape[0]):
        video_id = feature_df.iat[i,0]
        labels[i] = split_dict[video_id]['PHQ_Binary']
    feature_df['label'] = pd.Series(labels, index=feature_df.index)

    # Drop common (unwanted) columns - question, starttime, endtime
    try:
        feature_df.drop(['question','starttime','endtime'], inplace=True, axis=1)
    except ValueError:
        feature_df.drop(['question'], inplace=True, axis=1)
    return feature_df

def remove_low_variance(df):
    # Store feature names
    column_names=list(df.columns.values)

    # Obtain high variance features - indices
    X=df.as_matrix()
    sel = VarianceThreshold(0.95)
    sel.fit(X)
    selected_feature_idxs = sel.get_support(indices=True)

    # Obtain feature name list for indices
    selected_features = [column_names[i] for i in selected_feature_idxs]

    # Return data frame with selected features
    final_df = df[selected_features]
    return final_df

def perform_l1(df,labels):
    # Store feature names
    column_names=list(df.columns.values)

    # Obtain Selected features using L1 norm - indices
    X = df.as_matrix()
    y =labels
    svc = LogisticRegression(C=1., penalty='l1', dual=False).fit(X,y)
    model = SelectFromModel(svc,prefit=True)
    selected_feature_idxs = model.get_support(indices=True)

    # Obtain feature name list for indices
    selected_features = [column_names[i] for i in selected_feature_idxs]

    # Return data frame with selected features
    final_df = df[selected_features]
    return final_df

def select_best_K(df,labels,K):
    # Store feature names
    column_names=list(df.columns.values)

    # Obtain Selected K best features - indices
    X = df.as_matrix()
    y = labels
    kbest = SelectKBest(f_classif, k=K)
    kbest.fit(X, y)
    score_list = kbest.scores_

    # Sort feature indices from best to worst
    ind = [i for i in range(len(score_list))]
    all_pairs = []
    for a, b in zip(score_list, ind):
        all_pairs.append((a, b))
    sorted_pairs = sorted(all_pairs, key=lambda p: p[0], reverse=True)
    selected_feature_idxs = []
    for a, b in sorted_pairs:
        selected_feature_idxs.append(b)
    selected_feature_idxs = selected_feature_idxs[:K]

    # Obtain feature name list for indices
    selected_features = [column_names[i] for i in selected_feature_idxs]

    # Return data frame with selected features
    final_df = df[selected_features]
    return final_df

def perform_random_forest(df,labels,N):
    # Store feature names
    column_names=list(df.columns.values)

    # Obtain important features - indices
    X = df.as_matrix()
    y = labels
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(X, y)
    important_features = forest.feature_importances_

    # Sort feature indices from best to worst
    ind = [i for i in range(len(important_features))]
    all_pairs = []
    for a, b in zip(important_features, ind):
        all_pairs.append((a, b))
    sorted_pairs=sorted(all_pairs,key=lambda p:p[0],reverse=True)
    selected_feature_idxs = []
    for a, b in sorted_pairs:
        selected_feature_idxs.append(b)
    selected_feature_idxs = selected_feature_idxs[:N]

    # Obtain feature name list for indices
    selected_features = [column_names[i] for i in selected_feature_idxs]

    # Return data frame with selected features
    final_df = df[selected_features]
    return final_df

def main(qtype,mode):
    # Determine file name prefixes based on Question Type passed
    if qtype=="D":
        file_prefix="discriminative"
    elif qtype=="ND":
        file_prefix="nondiscriminative"
    elif qtype=="P":
        file_prefix="positive"
    else:
        file_prefix="negative"

    # Determine file_list based on Mode
    if mode == "V":
        file_list = ["_CLM.csv","_CLM_3D.csv","_CLM_Gaze.csv","_CLM_pose.csv","_FACET.csv"]
    elif mode == "A":
        file_list = ["_COVAREP.csv","_FORMANT.csv"]
    else:
        file_list = ["_LIWC.csv"]

    # Determine final file list for Question Type and Mode passed
    for i in range(len(file_list)):
        file_list[i] = file_prefix+file_list[i]
    print "File List: ",file_list

    # Determine directory
    if qtype=="D" or qtype=="ND":
        dir=config.D_ND_DIR
    else:
        dir=config.POS_NEG_DIR

    # Obtain file list with complete path
    file1 = os.path.join(dir,file_list[0])
    files = [os.path.join(dir,arg) for arg in file_list[1:]]

    # Obtain data frame containing all features from determined file list for TRAINING SET
    TRAIN = "y"
    df = get_feature_df(TRAIN,file1,files)

    # Obtain data frame containing all features from determined file list for VALIDATION SET
    TRAIN = "n"
    val_df = get_feature_df(TRAIN, file1, files)

    # If mode is visual, drop the extra columns from file - standardizes structure of data frame between all modes
    if mode=="V":
        df = df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)
        val_df = val_df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)

    # Obtain labels
    labels = df['label'].values

    # Make copy of data frame
    copy_df = df.copy() # copy_df contains values for - 'video', all features, 'label' columns

    # Remove 'video' and 'label' column from data frame
    df.drop(['video', 'label'], inplace=True , axis=1)

    # Pick 'N' to pick from Random Forest method, based on Mode
    if mode=="V":
        N = 100
    elif mode=="A":
        N = 20
    else:
        N = 50

    # Set 'K' to pick from Select Best K method
    K = 20

    # Call pipeline of feature selection methods on data frame - different pipeline for each Question Type and Mode combination
    if mode=="V":
        df = remove_low_variance(df)
    df = perform_l1(df,labels)
    df = perform_random_forest(df,labels,N)
    if mode!="A":
        df = select_best_K(df,labels,K)

    # Obtain Final feature list
    final_feature_list = list(df.columns.values)
    print "Final Feature List (Sorted): ",final_feature_list

    # Obtain data frame (for TRAIN and VALIDATION) to write into files
    final_selection = ['video']
    final_selection.extend(final_feature_list)
    final_selection.extend(['label'])
    op_df = copy_df[final_selection]
    op_val_df = val_df[final_selection]

    # To construct Output File Name
    if mode=="V":
        output_file="_visual"
    elif mode=="A":
        output_file="_acoustic"
    else:
        output_file="_linguistic"
    file_suffix_train="_train.csv"
    file_suffix_val="_val.csv"

    # Write output dfs into output files - TRAIN AND VALIDATION
    fileOP = os.path.join(config.SEL_FEAT_TRAIN,file_prefix + output_file + file_suffix_train)
    op_df.to_csv(fileOP,sep=",",index=False)
    fileOP = os.path.join(config.SEL_FEAT_VAL, file_prefix + output_file + file_suffix_val)
    op_val_df.to_csv(fileOP, sep=",", index=False)

    return final_feature_list

def feature_select():
    all_feature_lists = []

    # Call feature select function for all question types and modes
    question_types = ["D","ND","P","N"]
    modes = ["V","A","L"]
    for qtype in question_types:
        for mode in modes:
            print "Feature Selection for ",qtype," and ",mode
            feature_list = main(qtype,mode)
            all_feature_lists.append(feature_list)
    print "All features: ",all_feature_lists

    # Write all feature lists into output file
    file = os.path.join(config.SEL_FEAT, "all_selected_features.csv")
    fileOP = open(file,"w")
    for each_list in all_feature_lists:
        for feature in each_list:
            fileOP.write(feature)
            fileOP.write(",")
        fileOP.write("\n")

if __name__ == '__main__':
    #qtype = sys.argv[1] # D- discriminative, ND- nondiscriminative, P-positive, N- negative
    #mode = sys.argv[2] # A- acoustic, V- visual, L- linguistic

    # Call main function
    #main(qtype,mode)
    feature_select()