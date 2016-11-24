import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv
import os
import fnmatch
import sys

followUp = {}
ack = {}
nonIntimate = {}
intimate = {}
featureList = {}
questionType_DND={}
questionType_PN={}

def readHelperData():
    global followUp, ack, nonIntimate, intimate, questionType_PN, questionType_DND
    utterrances = pd.read_csv('data/misc/IdentifyingFollowUps.csv')
    disc_nondisc=pd.read_csv('data/misc/DND_Annotations.csv')
    pos_neg=pd.read_csv('data/misc/PN_Annotations.csv')

    #Discriminative/Non-discriminative annotations
    for i in xrange(len(disc_nondisc)):
        question=disc_nondisc.iloc[i]['Questions']
        qType=disc_nondisc.iloc[i]['Annotations']
        questionType_DND[question]=qType

    #Positive/Negative annotations
    for i in xrange(len(pos_neg)):
        question=pos_neg.iloc[i]['Questions']
        qType=pos_neg.iloc[i]['Annotations']
        questionType_PN[question]=qType

    for item in utterrances.itertuples():
        if item[3]=="#follow_up" and item[1] not in followUp:
            followUp[item[1]]=item[2]
        elif item[3]=="#ack" and item[1] not in ack:
            ack[item[1]]=item[2]
        elif item[3]=="#non_int" and item[1] not in nonIntimate:
            nonIntimate[item[1]]=item[2]
        elif item[3]=="#int" and item[1] not in intimate:
            intimate[item[1]]=item[2]

def readTranscript():
    global featureList
    transcriptFiles=glob(sys.argv[1]+'[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')
    for i in range(0,len(transcriptFiles)):
        t=pd.read_csv(transcriptFiles[i], delimiter='\t')
        t = t.fillna("")
        captureStarted=False
        startTime=0.0
        endTime=0.0
        prevQuestion=""
        participantNo=transcriptFiles[i][-18:-15]
        for j in xrange(len(t)):
            question=re.search(".*\((.*)\)$", t.iloc[j]['value'])
            if question is not None:
                question=question.group(1)
            else:
                question=t.iloc[j]['value']
            question=question.strip()

            if t.iloc[j]['speaker']=='Ellie':
                if question in nonIntimate and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)]=[startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1]=endTime
                    captureStarted=False

                elif question in intimate and question in questionType_DND and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)]=[startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1]=endTime
                    startTime=t.iloc[j]['start_time']
                    endTime=t.iloc[j]['stop_time']
                    prevQuestion=question

                elif question in intimate and question in questionType_DND and not captureStarted:
                    startTime=t.iloc[j]['start_time']
                    endTime=t.iloc[j]['stop_time']
                    prevQuestion=question
                    captureStarted=True

                elif question in intimate and question not in questionType_DND and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)]=[startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1]=endTime
                    captureStarted=False

                elif question in followUp or question in ack and captureStarted:
                    endTime=t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker']=='Participant' and captureStarted:
                endTime=t.iloc[j]['stop_time']





def readCLM_DND():
    groupByQuestion = {}

    dFile2 = open('data/disc_nondisc/discriminative_CLM_pose.csv', 'w')
    ndFile2 = open('data/disc_nondisc/nondiscriminative_CLM_pose.csv', 'w')

    dWriter2 = csv.writer(dFile2)
    ndWriter2 = csv.writer(ndFile2)
    header = ["video","question","starttime","endtime",'frame', 'timestamp', 'confidence', 'success', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    dWriter2.writerow(header)
    ndWriter2.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:

        fileName2 = sys.argv[1] + item + '_P/' + item + '_CLM_pose.txt'

        f2 = pd.read_csv(fileName2, delimiter=', ')

        for instance in groupByQuestion[item]:

            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f2.ix[(f2['timestamp'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f2.ix[(f2['timestamp'] - endTime).abs().argsort()[:1]].index.tolist()[0]
            #print startFrame, endFrame

            features = f2.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, instance[0])
            vector.insert(0, item)
            vector = np.asarray(vector)
            #print(item, instance[0], instance[1][1], instance[1][2])

            if questionType_DND[instance[0]] == 'D':
                dWriter2.writerow(vector)
            else:
                ndWriter2.writerow(vector)

def readCLM_PN():
    groupByQuestion = {}

    pFile2 = open('data/pos_neg/positive_CLM_pose.csv', 'w')
    nFile2 = open('data/pos_neg/negative_CLM_pose.csv', 'w')

    pWriter2 = csv.writer(pFile2)
    nWriter2 = csv.writer(nFile2)
    header = ["video","question","starttime","endtime",'frame', 'timestamp', 'confidence', 'success', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    pWriter2.writerow(header)
    nWriter2.writerow(header)

    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:

        fileName2 = sys.argv[1] + item + '_P/' + item + '_CLM_pose.txt'

        f2 = pd.read_csv(fileName2, delimiter=', ')

        for instance in groupByQuestion[item]:

            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f2.ix[(f2['timestamp'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f2.ix[(f2['timestamp'] - endTime).abs().argsort()[:1]].index.tolist()[0]
            #print startFrame, endFrame

            features = f2.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, instance[0])
            vector.insert(0, item)
            vector = np.asarray(vector)
            #print(item, instance[0], instance[1][1], instance[1][2])

            if questionType_PN[instance[0]] == 'P':
                pWriter2.writerow(vector)
            else:
                nWriter2.writerow(vector)




if __name__ == "__main__":
    readHelperData()
    readTranscript()
    readCLM_DND()
    readCLM_PN()