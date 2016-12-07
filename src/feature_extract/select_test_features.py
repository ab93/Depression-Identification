import numpy as np
from ..main import config
import pandas as pd
import os

def get_feature_df(train, file_, *files):
    # Set directory based on Train and Validation
    # if train == 'y':
    #     split_file = config.TRAIN_SPLIT_FILE
    # else:
    #     split_file = config.VAL_SPLIT_FILE

    split_file=config.TEST_SPLIT_FILE


    # Append file columns to a single data frame
    feature_df = pd.read_csv(file_)
    if len(files):
        for f in files[0]:
            print f
            feature_df = pd.concat([feature_df, pd.read_csv(f)], axis=1)
            feature_df = feature_df.T.drop_duplicates().T

    # Trim data frame to hole only train/validation records
    split_df = pd.read_csv(split_file,usecols=['Participant_ID', 'PHQ_Binary','PHQ_Score'])
    feature_df = feature_df[feature_df['video'].isin(split_df['Participant_ID'])]

    # Populate labels accordingly
    split_dict = split_df.set_index('Participant_ID').T.to_dict()
    del split_df
    labels = np.zeros(feature_df.shape[0])
    scores = np.zeros(feature_df.shape[0])
    for i in xrange(feature_df.shape[0]):
        video_id = feature_df.iat[i,0]
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


def main(qtype,mode,feature_list):
    # Determine file name prefixes based on Question Type passed
    if qtype=="P":
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
    dir=config.POS_NEG_DIR

    # Obtain file list with complete path
    file1 = os.path.join(dir,file_list[0])
    files = [os.path.join(dir,arg) for arg in file_list[1:]]

    # Obtain data frame containing all features from determined file list for TESTING SET
    TEST = "y"
    feature_list.append("video")
    feature_list.append("label")
    feature_list.append("score")
    df = get_feature_df(TEST,file1,files)
    df_new = df[feature_list]
    df_new.to_csv(config.SEL_FEAT)

def some_func():
    p_v = ['joyEvidence', 'AU9Evidence', 'confusionEvidence', 'isMaleEvidence', 'Z12', 'Z13', 'Z37', 'Z41', 'Z36',
           'Z11', 'Z19', 'Z14', 'Z44', 'Z3', 'Z45', 'Z52', 'Z51', 'AU43Evidence', 'Z5', 'Z27']
    p_a = ['F0', 'MCEP_7', 'MCEP_4', 'MCEP_8', 'MCEP_12', 'MCEP_17', 'HMPDD_2', 'MCEP_13', 'MCEP_11', 'MCEP_6',
           'MCEP_10', 'MCEP_5', 'MCEP_2', 'HMPDM_10', 'HMPDM_8', 'HMPDD_0', 'MCEP_0', 'Rd', 'formant1', 'HMPDM_11']
    p_l = ['avg_wordlen', 'coherence', 'word30', 'word83', 'word98', 'word54', 'word35', 'word66', 'word18', 'word28',
           'word5', 'word0', 'word52', 'word62', 'word32', 'word77', 'root_deps', 'word20', 'word53', 'word37']
    n_v = ['joyEvidence', 'Z21', 'Z20', 'Z19', 'Z37', 'Z38', 'Z40', 'Z18', 'Z1', 'Z2', 'isMaleEvidence', 'Z27', 'Z43',
           'Z4', 'Z44', 'Z47', 'AU43Evidence', 'Z13', 'Z28', 'Z51']
    n_a = ['MCEP_7', 'MCEP_8', 'MCEP_13', 'F0', 'MCEP_4', 'HMPDD_3', 'MCEP_6', 'MCEP_10', 'HMPDD_1', 'MCEP_3',
           'HMPDD_2', 'MCEP_2', 'formant4', 'formant5', 'formant1', 'HMPDM_9', 'MCEP_1', 'HMPDM_24', 'HMPDM_11',
           'HMPDM_10']
    n_l = ['word83', 'word49', 'word20', 'word35', 'word97', 'word42', 'word28', 'x_tag', 'word4', 'word92', 'word98',
           'word91', 'word75', 'word93', 'word60', 'word16', 'word79', 'word46', 'word82', 'word77']

    main("P","A",p_a)
    main("N", "A", n_a)
    main("P", "V", p_v)
    main("N", "V", n_v)
    main("P", "L", p_l)
    main("N", "L", n_l)