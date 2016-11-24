import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv
import sys

train={}
dev={}
features={}

x_d_acoustic, x_nd_acoustic, x_d_visual, x_nd_visual, x_d_linguistic, x_nd_linguistic={},{},{},{},{},{}
x_p_acoustic, x_n_acoustic, x_p_visual, x_n_visual, x_p_linguistic, x_n_linguistic={},{},{},{},{},{}
y_acoustic_train, y_visual_train, y_linguistic_train=[],[],[]
y_acoustic_dev, y_visual_dev, y_linguistic_dev=[],[],[]
x_d_acoustic_train, x_nd_acoustic_train, x_d_visual_train, x_nd_visual_train, x_d_linguistic_train, x_nd_linguistic_train=[],[],[],[],[],[]
x_p_acoustic_train, x_n_acoustic_train, x_p_visual_train, x_n_visual_train, x_p_linguistic_train, x_n_linguistic_train=[],[],[],[],[],[]
x_d_acoustic_dev, x_nd_acoustic_dev, x_d_visual_dev, x_nd_visual_dev, x_d_linguistic_dev, x_nd_linguistic_dev=[],[],[],[],[],[]
x_p_acoustic_dev, x_n_acoustic_dev, x_p_visual_dev, x_n_visual_dev, x_p_linguistic_dev, x_n_linguistic_dev=[],[],[],[],[],[]


def readLabels():
    train_data = pd.read_csv('data/classification_data/training_split.csv')
    dev_data = pd.read_csv('data/classification_data/dev_split.csv')
    #print train_data
    for i in xrange(len(train_data)):
        video=train_data.iloc[i]['Participant_ID']
        label=train_data.iloc[i]['PHQ_Binary']
        train[video]=label
    
    for i in xrange(len(dev_data)):
        video=dev_data.iloc[i]['Participant_ID']
        label=dev_data.iloc[i]['PHQ_Binary']
        dev[video]=(label)
    pprint(train)
    pprint(dev)

def read_data(reqd_features, feat):
    global features
    d=pd.read_csv('data/disc_nondisc/discriminative_'+feat+'.csv')
    nd=pd.read_csv('data/disc_nondisc/nondiscriminative_'+feat+'.csv')
    p=pd.read_csv('data/pos_neg/positive_'+feat+'.csv')
    n=pd.read_csv('data/pos_neg/negative_'+feat+'.csv')
    print reqd_features,feat
    d=d[reqd_features]
    nd=nd[reqd_features]
    p=p[reqd_features]
    n=n[reqd_features]

    d_x,nd_x,p_x,n_x=[],[],[],[]
    for name, group in d.groupby(["video"]):
        group=group[reqd_features]
        d_x.append(map(list,group.values))
    features[feat+'_dx']=d_x

    for name, group in nd.groupby(["video"]):
        group=group[reqd_features]
        nd_x.append(map(list,group.values))
    features[feat+'_ndx']=nd_x

    for name, group in p.groupby(["video"]):
        group=group[reqd_features]
        p_x.append(map(list,group.values))
    features[feat+'_px']=p_x

    for name, group in n.groupby(["video"]):
        group=group[reqd_features]
        n_x.append(map(list,group.values))
    features[feat+'_nx']=n_x

def process_acoustic(features_acoustic):
    global features
    features = {}
    read_data(features_acoustic[0],"COVAREP")
    read_data(features_acoustic[1],"FORMANT")
    #pprint(features)

    #discriminative features
    covarep=features['COVAREP_dx']
    formant=features['FORMANT_dx']

    for i in range(0,len(covarep)):
        for j in range(0,len(covarep[i])):
            if (covarep[i][j][0]) not in x_d_acoustic:
                x_d_acoustic[(covarep[i][j][0])]=[covarep[i][j][1:]+formant[i][j][1:]]
            else:
                x_d_acoustic[(covarep[i][j][0])].append(covarep[i][j][1:]+formant[i][j][1:])
    #pprint(x_d_acoustic)


    #non discriminative features
    covarep = features['COVAREP_ndx']
    formant = features['FORMANT_ndx']

    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):
            #print covarep[i][j][0]
            #raw_input()
            if covarep[i][j][0] not in x_nd_acoustic:
                x_nd_acoustic[covarep[i][j][0]] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_nd_acoustic[covarep[i][j][0]].append(covarep[i][j][1:] + formant[i][j][1:])
    #pprint(x_nd_acoustic)

    # positive features
    covarep = features['COVAREP_px']
    formant = features['FORMANT_px']
    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):

            if (covarep[i][j][0]) not in x_p_acoustic:
                x_p_acoustic[(covarep[i][j][0])] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_p_acoustic[(covarep[i][j][0])].append(covarep[i][j][1:] + formant[i][j][1:])
    #pprint(x_p_acoustic)

    # negative features
    covarep = features['COVAREP_nx']
    formant = features['FORMANT_nx']

    for i in range(0, len(covarep)):
        for j in range(0, len(covarep[i])):

            if (covarep[i][j][0]) not in x_n_acoustic:

                x_n_acoustic[(covarep[i][j][0])] = [covarep[i][j][1:] + formant[i][j][1:]]
            else:
                x_n_acoustic[(covarep[i][j][0])].append(covarep[i][j][1:] + formant[i][j][1:])
    #pprint(x_n_acoustic)

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
    #pprint(x_d_linguistic)

    # non discriminative features
    liwc = features['LIWC_ndx']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_nd_linguistic:
                x_nd_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_nd_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
    #pprint(x_nd_linguistic)


    # positive features
    liwc = features['LIWC_px']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_p_linguistic:
                x_p_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_p_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
    #pprint(x_p_linguistic)


    # negative features
    liwc = features['LIWC_nx']
    for i in range(0, len(liwc)):
        for j in range(0, len(liwc[i])):
            if liwc[i][j][0] not in x_n_linguistic:
                x_n_linguistic[liwc[i][j][0]] = [liwc[i][j][1:]]
            else:
                x_n_linguistic[liwc[i][j][0]].append(liwc[i][j][1:])
    #pprint(x_n_linguistic)


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
                x_d_visual[clm[i][j][0]] = [clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_d_visual[clm[i][j][0]].append(clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
    #pprint(x_d_visual)

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
    #pprint(x_nd_visual)

    # positive features
    clm = features['CLM_px']
    clm3d = features['CLM_3D_px']
    clm_gaze = features['CLM_Gaze_px']
    clm_pose = features['CLM_pose_px']
    facet = features['FACET_px']

    for i in range(0, len(clm)):
        for j in range(0, len(clm[i])):
            if clm[i][j][0] not in x_p_visual:
                x_p_visual[clm[i][j][0]] = [clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_p_visual[clm[i][j][0]].append(clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
    #pprint(x_p_visual)

    # negative features
    clm = features['CLM_nx']
    clm3d = features['CLM_3D_nx']
    clm_gaze = features['CLM_Gaze_nx']
    clm_pose = features['CLM_pose_nx']
    facet = features['FACET_nx']

    for i in range(0, len(clm)):
        for j in range(0, len(clm[i])):
            if clm[i][j][0] not in x_n_visual:
                x_n_visual[clm[i][j][0]] = [clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:]]
            else:
                x_n_visual[clm[i][j][0]].append(clm[i][j][1:] + clm3d[i][j][1:] + clm_gaze[i][j][1:] + clm_pose[i][j][1:] + facet[i][j][1:])
    #pprint(x_n_visual)

def create_x_y_matrix():

    for key in sorted(train.keys()):
        print key
        x_d_acoustic_train.append(key)
        x_nd_acoustic_train.append(key)
        x_p_acoustic_train.append(key)
        x_n_acoustic_train.append(key)

        y_acoustic_train.append(train[key] * len(x_d_acoustic[key][0]))

        x_d_visual_train.append(key)
        x_nd_visual_train.append(key)
        x_p_visual_train.append(key)
        x_n_visual_train.append(key)

        y_visual_train.append(train[key] * len(x_d_visual[key][0]))

        # x_d_linguistic_train.append(key)
        # x_nd_linguistic_train.append(key)
        # x_p_linguistic_train.append(key)
        # x_n_linguistic_train.append(key)
        #
        # y_linguistic_train.append(train[key] * len(x_d_linguistic[key][0]))

    for key in sorted(dev.keys()):
        print key
        x_d_acoustic_dev.append(key)
        x_nd_acoustic_dev.append(key)
        x_p_acoustic_dev.append(key)
        x_n_acoustic_dev.append(key)

        y_acoustic_dev.append([dev[key] * len(x_d_acoustic[key][0])])

        x_d_visual_dev.append(key)
        x_nd_visual_dev.append(key)
        x_p_visual_dev.append(key)
        x_n_visual_dev.append(key)

        y_visual_dev.append(dev[key] * len(x_d_visual[key][0]))
        #
        # x_d_linguistic_dev.append(key)
        # x_nd_linguistic_dev.append(key)
        # x_p_linguistic_dev.append(key)
        # x_n_linguistic_dev.append(key)
        #
        # y_linguistic_dev.append(dev[key] * len(x_d_linguistic[key][0]))

if __name__=="__main__":

    #please keep video in the featurelist!!!!
    #[[covarep],[formant]]

    features_acoustic=[["video",'F0','VUV','NAQ','QOQ'],["video","formant1","formant2"]]
    process_acoustic(features_acoustic)
    #[[clm],[clm3d],[clmgaze],[clmpose],[facet]]
    features_visual = [['video','x0',],['video','X0'],['video','x_0'],['video','Tx'],['video','Face X']]
    process_visual(features_visual)
    #[[LIWC]]
    features_linguistic = [['video','u_tag']]
    process_ling(features_linguistic)
    #pprint(x_d_acoustic)
    readLabels()
    create_x_y_matrix()
    print x_d_acoustic_dev
    print y_acoustic_dev



