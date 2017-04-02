import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv
import sys

followUp = {}
ack = {}
nonIntimate = {}
intimate = {}
featureList = {}
questionType_DND = {}
questionType_PN = {}

discriminativeVectors = []
nonDiscriminativeVectors = []

def readHelperData():
    global followUp, ack, nonIntimate, intimate, questionType_PN, questionType_DND
    utterrances = pd.read_csv('data/misc/IdentifyingFollowUps.csv')
    disc_nondisc = pd.read_csv('data/misc/DND_Annotations.csv')
    pos_neg = pd.read_csv('data/misc/PN_Annotations.csv')

    # Discriminative/Non-discriminative annotations
    for i in xrange(len(disc_nondisc)):
        question = disc_nondisc.iloc[i]['Questions']
        qType = disc_nondisc.iloc[i]['Annotations']
        questionType_DND[question] = qType

    # Positive/Negative annotations
    for i in xrange(len(pos_neg)):
        question = pos_neg.iloc[i]['Questions']
        qType = pos_neg.iloc[i]['Annotations']
        questionType_PN[question] = qType

    for item in utterrances.itertuples():
        if item[3] == "#follow_up" and item[1] not in followUp:
            followUp[item[1]] = item[2]
        elif item[3] == "#ack" and item[1] not in ack:
            ack[item[1]] = item[2]
        elif item[3] == "#non_int" and item[1] not in nonIntimate:
            nonIntimate[item[1]] = item[2]
        elif item[3] == "#int" and item[1] not in intimate:
            intimate[item[1]] = item[2]

def readTranscript():
    global featureList
    transcriptFiles = glob(sys.argv[1] + '[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')
    for i in range(0, len(transcriptFiles)):
        t = pd.read_csv(transcriptFiles[i], delimiter='\t')
        t = t.fillna("")
        captureStarted = False
        startTime = 0.0
        endTime = 0.0
        prevQuestion = ""
        participantNo = transcriptFiles[i][-18:-15]
        for j in xrange(len(t)):
            question = re.search(".*\((.*)\)$", t.iloc[j]['value'])
            if question is not None:
                question = question.group(1)
            else:
                question = t.iloc[j]['value']
            question = question.strip()

            if t.iloc[j]['speaker'] == 'Ellie':
                if question in nonIntimate and captureStarted:
                    endTime=t.iloc[j]['start_time']
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in intimate and question in questionType_DND and captureStarted:
                    endTime=t.iloc[j]['start_time']
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    startTime = t.iloc[j]['start_time']
                    endTime = t.iloc[j]['stop_time']
                    prevQuestion = question

                elif question in intimate and question in questionType_DND and not captureStarted:
                    startTime = t.iloc[j]['start_time']
                    endTime = t.iloc[j]['stop_time']
                    prevQuestion = question
                    captureStarted = True

                elif question in intimate and question not in questionType_DND and captureStarted:
                    endTime=t.iloc[j]['start_time']
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in followUp or question in ack and captureStarted:
                    endTime = t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker'] == 'Participant' and captureStarted:
                continue
                #endTime = t.iloc[j]['stop_time']

def readOPENFACE_DND():
    groupByQuestion = {}
    dFile = open('data/disc_nondisc/discriminative_OPENFACE.csv', 'w')
    ndFile = open('data/disc_nondisc/nondiscriminative_OPENFACE.csv', 'w')
    dWriter = csv.writer(dFile)
    ndWriter = csv.writer(ndFile)
    header = ["video", "question", "starttime", "endtime","frame","timestamp","confidence","success",
              "pose_Tx","pose_Ty","pose_Tz","pose_Rx","pose_Ry","pose_Rz","AU01_r","AU02_r","AU04_r","AU05_r",
              "AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r",
              "AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c",
              "AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]
    dWriter.writerow(header)
    ndWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_OPENFACE.txt'
        f = pd.read_csv(fileName, delimiter=', ')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f.ix[(f['timestamp'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f.ix[(f['timestamp'] - endTime).abs().argsort()[:1]].index.tolist()[0]
            features = f.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, instance[0])
            vector.insert(0, item)
            vector = np.asarray(vector)
            # print item, instance[0], startTime, endTime

            if questionType_DND[instance[0]] == 'D':
                dWriter.writerow(vector)
            else:
                ndWriter.writerow(vector)
    dFile.close()
    ndFile.close()


def readOPENFACE_PN():
    groupByQuestion = {}
    pFile = open('data/pos_neg/positive_OPENFACE.csv', 'w')
    nFile = open('data/pos_neg/negative_OPENFACE.csv', 'w')
    pWriter = csv.writer(pFile)
    nWriter = csv.writer(nFile)
    header = ["video", "question", "starttime", "endtime","frame","timestamp","confidence","success",
              "pose_Tx","pose_Ty","pose_Tz","pose_Rx","pose_Ry","pose_Rz","AU01_r","AU02_r","AU04_r","AU05_r",
              "AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r",
              "AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c",
              "AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]

    pWriter.writerow(header)
    nWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_OPENFACE.txt'
        f = pd.read_csv(fileName, delimiter=', ')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f.ix[(f['timestamp'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f.ix[(f['timestamp'] - endTime).abs().argsort()[:1]].index.tolist()[0]

            features = f.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, instance[0])
            vector.insert(0, item)
            vector = np.asarray(vector)
            # print item, instance[0], startTime, endTime

            if questionType_PN[instance[0]] == 'P':
                pWriter.writerow(vector)
            else:
                nWriter.writerow(vector)
    pFile.close()
    nFile.close()


if __name__ == "__main__":
    readHelperData()
    readTranscript()
    readCLM_DND()
    readCLM_PN()