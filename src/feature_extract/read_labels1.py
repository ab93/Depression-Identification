import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv
import sys
import config

train = {}
dev = {}
features = {}

x_d_acoustic, x_nd_acoustic, x_d_visual, x_nd_visual, x_d_linguistic, x_nd_linguistic = {}, {}, {}, {}, {}, {}
x_p_acoustic, x_n_acoustic, x_p_visual, x_n_visual, x_p_linguistic, x_n_linguistic = {}, {}, {}, {}, {}, {}
y_d_acoustic_train, y_nd_acoustic_train, y_d_visual_train, y_nd_visual_train, y_d_linguistic_train, y_nd_linguistic_train = [], [], [], [], [], []
y_p_acoustic_train, y_n_acoustic_train, y_p_visual_train, y_n_visual_train, y_p_linguistic_train, y_n_linguistic_train = [], [], [], [], [], []
y_p_acoustic_dev, y_n_acoustic_dev, y_p_visual_dev, y_n_visual_dev, y_p_linguistic_dev, y_n_linguistic_dev = [], [], [], [], [], []
y_d_acoustic_dev, y_nd_acoustic_dev, y_d_visual_dev, y_nd_visual_dev, y_d_linguistic_dev, y_nd_linguistic_dev = [], [], [], [], [], []

x_d_acoustic_train, x_nd_acoustic_train, x_d_visual_train, x_nd_visual_train, x_d_linguistic_train, x_nd_linguistic_train = [], [], [], [], [], []
x_p_acoustic_train, x_n_acoustic_train, x_p_visual_train, x_n_visual_train, x_p_linguistic_train, x_n_linguistic_train = [], [], [], [], [], []
x_d_acoustic_dev, x_nd_acoustic_dev, x_d_visual_dev, x_nd_visual_dev, x_d_linguistic_dev, x_nd_linguistic_dev = [], [], [], [], [], []
x_p_acoustic_dev, x_n_acoustic_dev, x_p_visual_dev, x_n_visual_dev, x_p_linguistic_dev, x_n_linguistic_dev = [], [], [], [], [], []


def readLabels():
    '''

    Returns
    -------

    '''
    train_data = pd.read_csv('data/classification_data/training_split.csv')
    dev_data = pd.read_csv('data/classification_data/dev_split.csv')
    for i in xrange(len(train_data)):
        video = train_data.iloc[i]['Participant_ID']
        label = train_data.iloc[i]['PHQ_Binary']
        train[video] = label

    for i in xrange(len(dev_data)):
        video = dev_data.iloc[i]['Participant_ID']
        label = dev_data.iloc[i]['PHQ_Binary']
        dev[video] = (label)


def read_data(reqd_features, feat):
    global features
    d = pd.read_csv('data/disc_nondisc/discriminative_' + feat + '.csv')
    nd = pd.read_csv('data/disc_nondisc/nondiscriminative_' + feat + '.csv')
    p = pd.read_csv('data/pos_neg/positive_' + feat + '.csv')
    n = pd.read_csv('data/pos_neg/negative_' + feat + '.csv')
    # print reqd_features,feat
    d = d[reqd_features]
    nd = nd[reqd_features]
    p = p[reqd_features]
    n = n[reqd_features]

    d_x, nd_x, p_x, n_x = [], [], [], []
    for name, group in d.groupby(["video"]):
        group = group[reqd_features]
        d_x.append(map(list, group.values))
    features[feat + '_dx'] = d_x

    for name, group in nd.groupby(["video"]):
        group = group[reqd_features]
        nd_x.append(map(list, group.values))
    features[feat + '_ndx'] = nd_x

    for name, group in p.groupby(["video"]):
        group = group[reqd_features]
        p_x.append(map(list, group.values))
    features[feat + '_px'] = p_x

    for name, group in n.groupby(["video"]):
        group = group[reqd_features]
        n_x.append(map(list, group.values))
    features[feat + '_nx'] = n_x


def process_acoustic(features_acoustic):
    global features
    features = {}
    read_data(features_acoustic[0], "COVAREP")
    read_data(features_acoustic[1], "FORMANT")
    # pprint(features)

    # discriminative features
    covarep = features['COVAREP_dx']
    formant = features['FORMANT_dx']

    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):
            if (covarep[i][j][0]) not in x_d_acoustic:
                x_d_acoustic[(covarep[i][j][0])] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_d_acoustic[covarep[i][j][0]].append(covarep[i][j][1:] + formant[i][j][1:])

    # non discriminative features
    covarep = features['COVAREP_ndx']
    formant = features['FORMANT_ndx']

    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):
            if covarep[i][j][0] not in x_nd_acoustic:
                x_nd_acoustic[covarep[i][j][0]] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_nd_acoustic[covarep[i][j][0]].append(covarep[i][j][1:] + formant[i][j][1:])

    # positive features
    covarep = features['COVAREP_px']
    formant = features['FORMANT_px']
    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):
            if (covarep[i][j][0]) not in x_p_acoustic:
                x_p_acoustic[(covarep[i][j][0])] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_p_acoustic[(covarep[i][j][0])].append(covarep[i][j][1:] + formant[i][j][1:])

    # negative features
    covarep = features['COVAREP_nx']
    formant = features['FORMANT_nx']

    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):
            if (covarep[i][j][0]) not in x_n_acoustic:
                x_n_acoustic[(covarep[i][j][0])] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_n_acoustic[(covarep[i][j][0])].append(covarep[i][j][1:] + formant[i][j][1:])
                # pprint(x_n_acoustic)


def process_ling(features_linguistic):
    global features
    features = {}
    read_data(features_linguistic[0], "LIWC")
    # discriminative features
    liwc = features['LIWC_dx']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_d_linguistic:
                x_d_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_d_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
    # pprint(x_d_linguistic)

    # non discriminative features
    liwc = features['LIWC_ndx']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_nd_linguistic:
                x_nd_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_nd_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
    # pprint(x_nd_linguistic)


    # positive features
    liwc = features['LIWC_px']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_p_linguistic:
                x_p_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_p_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
    # pprint(x_p_linguistic)


    # negative features
    liwc = features['LIWC_nx']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_n_linguistic:
                x_n_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_n_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
                # pprint(x_n_linguistic)

def process(mode):
    #feature_df = pd.read_csv(file_)
    if mode == "visual":
        files = config.D_ND_DIR
    elif mode == "linguistic":
        files = config.D_ND_DIR
    else:
        files = config.D_ND_DIR
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
def process_visual(features_visual):
    global features
    features = {}
    read_data(features_visual[0], "CLM")
    read_data(features_visual[1], "CLM_3D")
    read_data(features_visual[2], "CLM_Gaze")
    read_data(features_visual[3], "CLM_pose")
    read_data(features_visual[4], "FACET")

    # pprint(features)

    # discriminative features
    clm = features['CLM_dx']
    clm3d = features['CLM_3D_dx']
    clm_gaze = features['CLM_Gaze_dx']
    clm_pose = features['CLM_pose_dx']
    facet = features['FACET_dx']

    for i in range(0, len(clm)):
        for j in range(0, len(clm[i])):
            if clm[i][j][0] not in x_d_visual:
                x_d_visual[clm[i][j][0]] = [
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_d_visual[clm[i][j][0]].append(
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
    # pprint(x_d_visual)

    # non discriminative features
    clm = features['CLM_ndx']
    clm3d = features['CLM_3D_ndx']
    clm_gaze = features['CLM_Gaze_ndx']
    clm_pose = features['CLM_pose_ndx']
    facet = features['FACET_ndx']

    for i in range(0, len(clm)):
        for j in range(0, len(clm[i])):
            if clm[i][j][0] not in x_nd_visual:
                x_nd_visual[clm[i][j][0]] = [
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_nd_visual[clm[i][j][0]].append(
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
    # pprint(x_nd_visual)

    # positive features
    clm = features['CLM_px']
    clm3d = features['CLM_3D_px']
    clm_gaze = features['CLM_Gaze_px']
    clm_pose = features['CLM_pose_px']
    facet = features['FACET_px']

    for i in range(0, len(clm)):
        for j in range(0, len(clm[i])):
            if clm[i][j][0] not in x_p_visual:
                x_p_visual[clm[i][j][0]] = [
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_p_visual[clm[i][j][0]].append(
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
    # pprint(x_p_visual)

    # negative features
    clm = features['CLM_nx']
    clm3d = features['CLM_3D_nx']
    clm_gaze = features['CLM_Gaze_nx']
    clm_pose = features['CLM_pose_nx']
    facet = features['FACET_nx']

    for i in range(0, len(clm)):
        for j in range(0, len(clm[i])):
            if clm[i][j][0] not in x_n_visual:
                x_n_visual[clm[i][j][0]] = [
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_n_visual[clm[i][j][0]].append(
                    clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
                # pprint(x_n_visual)


def acous_train():
    for key in sorted(train.keys()):
        if key in x_d_acoustic:
            x_d_acoustic_train.append(x_d_acoustic[key])
            temp = []
            for i in range(len(x_d_acoustic[key])):
                temp.append(train[key])
            y_d_acoustic_train.append(temp)
        if key in x_nd_acoustic:
            x_nd_acoustic_train.append(x_nd_acoustic[key])
            temp = []
            for i in range(len(x_nd_acoustic[key])):
                # print "print ",[[train[key]] * len(x_nd_acoustic[key][i])]
                temp.append(train[key])
            y_nd_acoustic_train.append(temp)
        if key in x_p_acoustic:
            x_p_acoustic_train.append(x_p_acoustic[key])
            temp = []
            for i in range(len(x_p_acoustic[key])):
                # print "print ",[[train[key]] * len(x_p_acoustic[key][i])]
                temp.append(train[key])
            y_p_acoustic_train.append(temp)
        if key in x_n_acoustic:
            x_n_acoustic_train.append(x_n_acoustic[key])
            temp = []
            for i in range(len(x_n_acoustic[key])):
                # print "print ",[[train[key]] * len(x_n_acoustic[key][i])]
                temp.append(train[key])
            y_n_acoustic_train.append(temp)


def acous_dev():
    for key in sorted(dev.keys()):

        if key in x_d_acoustic:
            x_d_acoustic_dev.append(x_d_acoustic[key])
            temp = []
            for i in range(len(x_d_acoustic[key])):
                temp.append(dev[key])
            y_d_acoustic_dev.append(temp)
        if key in x_nd_acoustic:
            x_nd_acoustic_dev.append(x_nd_acoustic[key])
            temp = []
            for i in range(len(x_nd_acoustic[key])):
                temp.append(dev[key])
            y_nd_acoustic_dev.append(temp)
        if key in x_p_acoustic:
            x_p_acoustic_dev.append(x_p_acoustic[key])
            temp = []
            for i in range(len(x_p_acoustic[key])):
                temp.append(dev[key])
            y_p_acoustic_dev.append(temp)
        if key in x_n_acoustic:
            x_n_acoustic_dev.append(x_n_acoustic[key])
            temp = []
            for i in range(len(x_n_acoustic[key])):
                temp.append(dev[key])
            y_n_acoustic_dev.append(temp)


def visual_train():
    for key in sorted(train.keys(), ):

        if key in x_d_visual and key in x_nd_visual:
            x_d_visual_train.append(x_d_visual[key])
            temp = []
            for i in range(len(x_d_visual[key])):
                # print "print ",[[train[key]] * len(x_d_visual[key][i])]
                temp.append(train[key])
            y_d_visual_train.append(temp)
        if key in x_nd_visual:
            x_nd_visual_train.append(x_nd_visual[key])
            temp = []
            for i in range(len(x_nd_visual[key])):
                # print "print ",[[train[key]] * len(x_nd_visual[key][i])]
                temp.append(train[key])
            y_nd_visual_train.append(temp)
        if key in x_p_visual:
            x_p_visual_train.append(x_p_visual[key])
            temp = []
            for i in range(len(x_p_visual[key])):
                # print "print ",[[train[key]] * len(x_p_visual[key][i])]
                temp.append(train[key])
            y_p_visual_train.append(temp)
        if key in x_n_visual:
            x_n_visual_train.append(x_n_visual[key])
            temp = []
            for i in range(len(x_n_visual[key])):
                # print "print ",[[train[key]] * len(x_n_visual[key][i])]
                temp.append(train[key])
            y_n_visual_train.append(temp)


def visual_dev():
    for key in sorted(dev.keys()):

        if key in x_d_visual:
            x_d_visual_dev.append(x_d_visual[key])
            temp = []
            for i in range(len(x_d_visual[key])):
                temp.append(dev[key])
            y_d_visual_dev.append(temp)
        if key in x_nd_visual:
            x_nd_visual_dev.append(x_nd_visual[key])
            temp = []
            for i in range(len(x_nd_visual[key])):
                temp.append(dev[key])
            y_nd_visual_dev.append(temp)
        if key in x_p_visual:
            x_p_visual_dev.append(x_p_visual[key])
            temp = []
            for i in range(len(x_p_visual[key])):
                temp.append(dev[key])
            y_p_visual_dev.append(temp)
        if key in x_n_visual:
            x_n_visual_dev.append(x_n_visual[key])
            temp = []
            for i in range(len(x_n_visual[key])):
                temp.append(dev[key])
            y_n_visual_dev.append(temp)


def linguistic_train():
    for key in sorted(train.keys()):

        if (key in x_d_linguistic) and (key in x_nd_linguistic):
            x_d_linguistic_train.append(x_d_linguistic[key])
            x_nd_linguistic_train.append(x_nd_linguistic[key])
            temp_d = []
            temp_nd = []
            for i in range(len(x_d_linguistic[key])):
                temp_d.append(train[key])
            for i in range(len(x_nd_linguistic[key])):
                temp_nd.append(train[key])
            y_d_linguistic_train.append(temp_d)
            y_nd_linguistic_train.append(temp_nd)
        # if key in x_nd_linguistic:
        #     x_nd_linguistic_train.append(x_nd_linguistic[key])
        #     temp = []
        #     for i in range(len(x_nd_linguistic[key])):
        #         temp.append(train[key])
        #     y_nd_linguistic_train.append(temp)
        if key in (x_p_linguistic) and (key in x_n_linguistic):
            x_p_linguistic_train.append(x_p_linguistic[key])
            x_n_linguistic_train.append(x_n_linguistic[key])
            temp_p = []
            temp_n = []
            for i in range(len(x_p_linguistic[key])):
                temp_p.append(train[key])
            y_p_linguistic_train.append(temp_p)
            for i in range(len(x_n_linguistic[key])):
                temp_n.append(train[key])
            y_n_linguistic_train.append(temp_n)
            # if key in x_n_linguistic:
            #     x_n_linguistic_train.append(x_n_linguistic[key])
            #     temp = []
            #     for i in range(len(x_n_linguistic[key])):
            #         temp.append(train[key])
            #     y_n_linguistic_train.append(temp)


def linguistic_dev():
    for key in sorted(dev.keys()):

        if (key in x_d_linguistic) and (key in x_nd_linguistic_dev):
            x_d_linguistic_dev.append(x_d_linguistic[key])
            x_nd_linguistic_dev.append(x_nd_linguistic[key])
            temp_d = []
            temp_nd = []
            for i in range(len(x_d_linguistic[key])):
                temp_d.append(dev[key])
            y_d_linguistic_dev.append(temp_d)
            for i in range(len(x_d_linguistic[key])):
                temp_nd.append(dev[key])
            y_nd_linguistic_dev.append(temp_nd)
        # if key in x_nd_linguistic:
        #     x_nd_linguistic_dev.append(x_nd_linguistic[key])
        #     temp = []
        #     for i in range(len(x_nd_linguistic[key])):
        #         temp.append(dev[key])
        #     y_nd_linguistic_dev.append(temp)
        if (key in x_p_linguistic) and (key in x_n_linguistic):
            x_p_linguistic_dev.append(x_p_linguistic[key])
            x_n_linguistic_dev.append(x_n_linguistic[key])
            temp_p = []
            temp_n = []
            for i in range(len(x_p_linguistic[key])):
                temp_p.append(dev[key])
            y_p_linguistic_dev.append(temp_p)
            for i in range(len(x_n_linguistic[key])):
                temp_n.append(dev[key])
            y_n_linguistic_dev.append(temp_n)



            # if key in x_n_linguistic:
            #     x_n_linguistic_dev.append(x_n_linguistic[key])
            #     temp = []
            #     for i in range(len(x_n_linguistic[key])):
            #         temp.append(dev[key])
            #     y_n_linguistic_dev.append(temp)


# features_acoustic = [["video", 'F0', 'VUV', 'NAQ', 'QOQ'], ["video", "formant1", "formant2"]]
def return_acou_dnd(features_acoustic):
    for i in features_acoustic:
        i.insert(0, "video")
    readLabels()
    # [[covarep],[formant]]
    process_acoustic(features_acoustic)
    acous_train()
    acous_dev()
    return x_d_acoustic_train, y_d_acoustic_train, x_nd_acoustic_train, y_nd_acoustic_train, x_d_acoustic_dev, y_d_acoustic_dev, x_nd_acoustic_dev, y_nd_acoustic_dev


def return_acou_pn(features_acoustic):
    for i in features_acoustic:
        i.insert(0, "video")
    readLabels()
    process_acoustic(features_acoustic)
    acous_train()
    acous_dev()
    return x_p_acoustic_train, y_p_acoustic_train, x_n_acoustic_train, y_n_acoustic_train, x_p_acoustic_dev, y_p_acoustic_dev, x_n_acoustic_dev, y_n_acoustic_dev


def return_vis_dnd(features_visual):
    for i in features_visual:
        i.insert(0, "video")
    readLabels()
    # [[clm],[clm3d],[clmgaze],[clmpose],[facet]]
    # features_visual = [['video', 'x0', ], ['video', 'X0'], ['video', 'x_0'], ['video', 'Tx'], ['video', 'Face X']]
    process_visual(features_visual)
    visual_train()
    visual_dev()
    return x_d_visual_train, y_d_visual_train, x_nd_visual_train, y_nd_visual_train, x_d_visual_dev, y_d_visual_dev, x_nd_visual_dev, y_nd_visual_dev


def return_vis_pn(features_visual):
    for i in features_visual:
        i.insert(0, "video")
    readLabels()
    # [[clm],[clm3d],[clmgaze],[clmpose],[facet]]
    # features_visual = [['video', 'x0', ], ['video', 'X0'], ['video', 'x_0'], ['video', 'Tx'], ['video', 'Face X']]
    process_visual(features_visual)
    visual_train()
    visual_dev()
    return x_p_visual_train, y_p_visual_train, x_n_visual_train, y_n_visual_train, x_p_visual_dev, y_p_visual_dev, x_n_visual_dev, y_n_visual_dev


def return_lin_dnd(features_linguistic):
    for i in features_linguistic:
        i.insert(0, "video")
    readLabels()
    # [[LIWC]]
    # features_linguistic = [['video', 'u_tag']]
    process_ling(features_linguistic)
    linguistic_train()
    linguistic_dev()
    return x_d_linguistic_train, y_d_linguistic_train, x_nd_linguistic_train, y_nd_linguistic_train, x_d_linguistic_dev, y_d_linguistic_dev, x_nd_linguistic_dev, y_nd_linguistic_dev


def return_lin_pn(features_linguistic):
    for i in features_linguistic:
        i.insert(0, "video")
    readLabels()
    # [[LIWC]]
    # features_linguistic = [['video', 'u_tag']]
    process_ling(features_linguistic)
    linguistic_train()
    linguistic_dev()
    return x_p_linguistic_train, y_p_linguistic_train, x_n_linguistic_train, y_n_linguistic_train, x_p_linguistic_dev, y_p_linguistic_dev, x_n_linguistic_dev, y_n_linguistic_dev

# pprint(return_acou_dev())

print config.D_ND_DIR