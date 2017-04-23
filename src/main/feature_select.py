import os
import numpy as np
import pandas as pd
import sys
import config
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

def get_feature_df(train, file_, *files):
    '''Obtain feature df of train/validation/test split

    Parameters
    ----------
    train : str, {'train','val','test'}
    file_ : str, Path to first feature file
    files : array[str], Paths to more feature files
    
    Returns
    -------
    feature_df : dataframe, Features obtained from input files based on split
    '''

    # Set split_file based on input - 'train'/'val'/'test'
    if train == 'train':
        split_file = config.TRAIN_SPLIT_FILE
    elif train == "val":
        split_file = config.VAL_SPLIT_FILE
    else:
        split_file = config.TEST_SPLIT_FILE

    # Append feature file columns to a single feature data frame
    feature_df = pd.read_csv(file_,error_bad_lines=False)
    feature_df = feature_df.fillna(0)
    if len(files):
        for f in files[0]:
            feature_df_second = pd.read_csv(f)
            feature_df_second = feature_df_second.fillna(0)
            feature_df = pd.concat([feature_df, feature_df_second], axis=1)
            feature_df = feature_df.T.drop_duplicates().T

    # Trim data frame to hold only train/validation records based on split_file
    if train == "test":
        split_df = pd.read_csv(split_file, usecols=['participant_ID'])
        feature_df = feature_df[feature_df['video'].isin(split_df['participant_ID'])]
    else:
        split_df = pd.read_csv(split_file,usecols=['Participant_ID', 'PHQ_Binary','PHQ_Score'])
        feature_df = feature_df[feature_df['video'].isin(split_df['Participant_ID'])]

        # Populate labels and scores accordingly from split_df
        split_dict = split_df.set_index('Participant_ID').T.to_dict()
        del split_df
        labels = np.zeros(feature_df.shape[0])
        scores = np.zeros(feature_df.shape[0])
        for i in xrange(feature_df.shape[0]):
            video_id = feature_df.iat[i, 0]
            labels[i] = split_dict[video_id]['PHQ_Binary']
            scores[i] = split_dict[video_id]['PHQ_Score']
        feature_df['label'] = pd.Series(labels, index=feature_df.index)
        feature_df['score'] = pd.Series(scores, index=feature_df.index)

    # Drop common (unwanted) columns - question, starttime, endtime
    try:
        feature_df.drop(['question','starttime','endtime'], inplace=True, axis=1)
    except ValueError:
        feature_df.drop(['question'], inplace=True, axis=1)

    return feature_df

def remove_low_variance(df):
    '''Remove low variance features

    Parameters
    ----------
    df : dataframe, Features to select from
    
    Returns
    -------
    final_df : dataframe, Features with only selected feature columns
    '''

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
    '''Perform L1 norm feature selection

    Parameters
    ----------
    df : dataframe, Features to select from
    labels : array-like, Labels/scores to select based on
    
    Returns
    -------
    final_df : dataframe, Features with selected feature columns
    '''

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
    '''Obtain K bst features

    Parameters
    ----------
    df : dataframe, Features to select from
    labels : array-like, Labels/scores to select based on
    K : int, Number of features to select
    
    Returns
    -------
    final_df : dataframe, Features with selected feature columns
    '''

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
    '''Perform Random Forest feature selection

    Parameters
    ----------
    df : dataframe, Features to select from
    labels : array-like, Labels/scores to select based on
    N : int, Number of features to select
    
    Returns
    -------
    final_df : dataframe, Features with selected feature columns
    '''

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

def main(qtype,mode,classifier_type):
    '''Performs feature selection for given category and mode

    Parameters
    ----------
    qtype : str, {'D','ND','P','N'}
    mode : str, {'A','V','L'}
    classifier_type : str, {'C','R'}
    
    Returns
    -------
    final_feature_list : array-like, Selected features' names
    '''

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
        file_list = ["_OPENFACE.csv"]
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
    TRAIN = "train"
    df = get_feature_df(TRAIN,file1,files)

    # Obtain data frame containing all features from determined file list for VALIDATION SET
    TRAIN = "val"
    val_df = get_feature_df(TRAIN,file1, files)

    # Obtain data frame containing all features from determined file list for TEST SET
    TRAIN = "test"
    test_df = get_feature_df(TRAIN,file1,files)

    '''
    # If mode is visual, drop the extra columns from file - standardizes structure of data frame between all modes
    if mode=="V":
        df = df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)
        val_df = val_df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)
        test_df = test_df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)
    '''

    # Create selection_df containing features from train and validation to perform feature selection on
    selection_df = pd.concat([df,val_df])

    # Obtain labels
    labels = selection_df['label'].values
    scores = selection_df['score'].values

    # Remove 'video' and 'label' column from selection_df
    selection_df.drop(['video', 'label','score'], inplace=True , axis=1)

    # Pick 'N' to pick from Random Forest method, based on Mode
    if mode=="A":
        N = 20
    else:
        N = 50

    # Set 'K' to pick from Select Best K method
    K = 20

    # Set feature_type to contain labels or scores based on classifier_type
    if(classifier_type == "C"):
        feature_type = labels
    elif(classifier_type == "R"):
        feature_type = scores

    # Call pipeline of feature selection methods on data frame - different pipeline for each Question Type and Mode combination
    if mode=="V":   
        selection_df = perform_random_forest(selection_df,feature_type,N)
    elif mode == "A":
        selection_df = perform_l1(selection_df,feature_type)
        selection_df = perform_random_forest(selection_df,feature_type,N)
    else:
        selection_df = perform_random_forest(selection_df,feature_type,N)
        selection_df = select_best_K(selection_df,feature_type,K)

    # Obtain Final feature list
    final_feature_list = list(selection_df.columns.values)
    print "Final Feature List (Sorted): ",final_feature_list

    # Obtain data frame (for TRAIN, VALIDATION AND TEST) to write into files
    final_selection = ['video']
    final_selection.extend(final_feature_list)
    final_selection.extend(['label'])
    final_selection.extend(['score'])
    op_df = df[final_selection]
    op_val_df = val_df[final_selection]
    final_selection.remove('label')
    final_selection.remove('score')
    op_test_df = test_df[final_selection]

    # To construct Output File Name
    if mode=="V":
        output_file="_visual"
    elif mode=="A":
        output_file="_acoustic"
    else:
        output_file="_linguistic"
    file_suffix_train="_train.csv"
    file_suffix_val="_val.csv"
    file_suffix_test = "_test.csv"

    # Write output dfs into output files - TRAIN, VALIDATION AND TEST
    if classifier_type == "C":
        directory_path_train = config.SEL_FEAT_TRAIN_REGULAR_CLASSIFY
        directory_path_val = config.SEL_FEAT_VAL_REGULAR_CLASSIFY
        directory_path_test = config.SEL_FEAT_TEST_REGULAR_CLASSIFY
    else:
        directory_path_train = config.SEL_FEAT_TRAIN_REGULAR_ESTIMATE
        directory_path_val = config.SEL_FEAT_VAL_REGULAR_ESTIMATE
        directory_path_test = config.SEL_FEAT_TEST_REGULAR_ESTIMATE
    fileOP = os.path.join(directory_path_train,file_prefix + output_file + file_suffix_train)
    op_df.to_csv(fileOP,sep=",",index=False)
    fileOP = os.path.join(directory_path_val, file_prefix + output_file + file_suffix_val)
    op_val_df.to_csv(fileOP, sep=",", index=False)
    fileOP = os.path.join(directory_path_test, file_prefix + output_file + file_suffix_test)
    op_test_df.to_csv(fileOP, sep=",", index=False)

    return final_feature_list

def feature_select(classifier_type):
    '''Calls feature selection method for all categories and modes, with given classifier_type

    Parameters
    ----------
    classifier_type : str, {'C','R'}
    '''

    all_feature_lists = []

    # Call feature select function for all question types and modes
    question_types = ["D","ND","P","N"]
    modes = ["V","A","L"]
    for qtype in question_types:
        for mode in modes:
            print "Feature Selection for ",qtype," and ",mode
            feature_list = main(qtype,mode,classifier_type)
            all_feature_lists.append(feature_list)
    print "All features: ",all_feature_lists

    # Write all feature lists into output file
    file = os.path.join(config.SEL_FEAT, classifier_type+"_all_selected_features.csv")
    fileOP = open(file,"w")
    for each_list in all_feature_lists:
        for feature in each_list:
            fileOP.write(feature)
            fileOP.write(",")
        fileOP.write("\n")

if __name__ == '__main__':
    # Call feature select function for both classification and regression
    feature_select("C")
    feature_select("R")