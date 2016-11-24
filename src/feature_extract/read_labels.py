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
x_p_acoustic, x_n_acoustic, x_p_visual, x_n_visual, x_p_linguistic, x_n_linguistic=[],[],[],[],[],[]
y_d_acoustic, y_nd_acoustic, y_d_visual, y_nd_visual, y_d_linguistic, y_nd_linguistic=[],[],[],[],[],[]
y_p_acoustic, y_n_acoustic, y_p_visual, y_n_visual, y_p_linguistic, y_n_linguistic=[],[],[],[],[],[]

d_facet_x, nd_facet_x, p_facet_x, n_facet_x=[],[],[],[]

def readLabels():
    train = pd.read_csv('data/classification_data/training_split.csv')
    dev = pd.read_csv('data/classification_data/dev_split.csv')

    for i in xrange(len(train)):
        video=train.iloc[i]['Participant_ID']
        label=train.iloc[i]['PHQ_Binary']
        train[video]=int(label)
    
    for i in xrange(len(train)):
        video=dev.iloc[i]['Participant_ID']
        label=dev.iloc[i]['PHQ_Binary']
        dev[video]=int(label)

def read_data(reqd_features, feat):
    global features
    d=pd.read_csv('data/disc_nondisc/discriminative_'+feat+'.csv')
    nd=pd.read_csv('data/disc_nondisc/nondiscriminative_'+feat+'.csv')
    p=pd.read_csv('data/pos_neg/positive_'+feat+'.csv')
    n=pd.read_csv('data/pos_neg/negative_'+feat+'.csv')
    
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

#FACET_DX, CLM_DX...
def merge_data():
    acoustic=['COVAREP','FORMANT']
    visual=['FACET','CLM','CLM_3D','CLM_GAZE','CLM_POSE']
    linguistic=['LIWC']

    facet=features['FACET_dx']
    clm=features['CLM_dx']

    for i in range(0,len(facet)):
        for j in range(0,len(facet[i])):
            if facet[i][j][0] not in x_d_acoustic:
                x_d_acoustic[facet[i][j][0]]=[facet[i][j][1:]+clm[i][j][1:]]
            else:
                x_d_acoustic[facet[i][j][0]].append([facet[i][j][1:]+clm[i][j][1:]])


if __name__=="__main__":
    header=["video","Face X","Face Y","Face Width","Face Height"]
    read_data(header, "FACET")
    pprint(features)