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
header = ["video","question","starttime","endtime",'F0','VUV','NAQ','QOQ','H1H2','PSP','MDQ','peakSlope','Rd',
          'Rd_conf','creak','MCEP_0','MCEP_1','MCEP_2','MCEP_3','MCEP_4','MCEP_5',
        'MCEP_6','MCEP_7','MCEP_8','MCEP_9','MCEP_10','MCEP_11','MCEP_12',
    'MCEP_13','MCEP_14','MCEP_15','MCEP_16','MCEP_17','MCEP_18',
    'MCEP_19','MCEP_20','MCEP_21','MCEP_22','MCEP_23','MCEP_24',
    'HMPDM_0','HMPDM_1','HMPDM_2','HMPDM_3','HMPDM_4','HMPDM_5',
    'HMPDM_6','HMPDM_7','HMPDM_8','HMPDM_9','HMPDM_10','HMPDM_11','HMPDM_12',
    'HMPDM_13','HMPDM_14','HMPDM_15','HMPDM_16','HMPDM_17','HMPDM_18',
    'HMPDM_19','HMPDM_20','HMPDM_21','HMPDM_22','HMPDM_23','HMPDM_24',
    'HMPDD_0','HMPDD_1','HMPDD_2','HMPDD_3','HMPDD_4','HMPDD_5',
    'HMPDD_6','HMPDD_7','HMPDD_8','HMPDD_9','HMPDD_10','HMPDD_11','HMPDD_12']
header_f = ["video","question","starttime","endtime","formant1",'formant2','formant3','formant4',"formant5"]

questionType_DND={}
questionType_PN={}
questionAnswers = {}


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
        t=pd.read_csv(transcriptFiles[i], delimiter=',|\t')
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
                endTime=t.iloc[j]['start_time']
                if question in nonIntimate and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)]=[startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1]=endTime
                    captureStarted=False

                elif question in intimate and question in questionType_DND and captureStarted:
                    endTime=t.iloc[j]['start_time']
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
                    endTime=t.iloc[j]['start_time']
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)]=[startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1]=endTime
                    captureStarted=False

                elif question in followUp or question in ack and captureStarted:
                    endTime=t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker']=='Participant' and captureStarted:
                #endTime=t.iloc[j]['stop_time']
                continue


def readFORMANT_DND():
    groupByQuestion = {}
    dFile = open('data/disc_nondisc/discriminative_FORMANT.csv', 'w')
    ndFile = open('data/disc_nondisc/nondiscriminative_FORMANT.csv', 'w')
    dWriter = csv.writer(dFile)
    ndWriter = csv.writer(ndFile)
    dWriter.writerow(header_f)
    ndWriter.writerow(header_f)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_FORMANT.csv'
        f = pd.read_csv(fileName, delimiter=',')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100
            # for i in xrange(len(f)):
            #     if f.ix[i]['Frametime'] >= startTime and startFrame == 0:
            #         startFrame = i
            #     elif f.ix[i]['Frametime'] >= endTime and endFrame == 0:
            #         endFrame = i - 1
            #print instance

            features = f.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, item)
            vector.insert(1, instance[0])
            vector = np.asarray(vector)
            if questionType_DND[instance[0]] == 'D':
                dWriter.writerow(vector)
            else:
                ndWriter.writerow(vector)

def readFORMANT_PN():
    groupByQuestion = {}
    pFile = open('data/pos_neg/positive_FORMANT.csv', 'w')
    nFile = open('data/pos_neg/negative_FORMANT.csv', 'w')
    pWriter = csv.writer(pFile)
    nWriter = csv.writer(nFile)
    pWriter.writerow(header_f)
    nWriter.writerow(header_f)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_FORMANT.csv'
        f = pd.read_csv(fileName, delimiter=',')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100
            # for i in xrange(len(f)):
            #     if f.ix[i]['Frametime'] >= startTime and startFrame == 0:
            #         startFrame = i
            #     elif f.ix[i]['Frametime'] >= endTime and endFrame == 0:
            #         endFrame = i - 1
            #print instance

            features = f.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, item)
            vector.insert(1, instance[0])
            vector = np.asarray(vector)
            if questionType_PN[instance[0]] == 'P':
                pWriter.writerow(vector)
            else:
                nWriter.writerow(vector)


def readCOVAREP_DND():
    groupByQuestion = {}
    dFile = open('data/disc_nondisc/discriminative_COVAREP.csv', 'a')
    ndFile = open('data/disc_nondisc/nondiscriminative_COVAREP.csv', 'a')
    dWriter = csv.writer(dFile)
    ndWriter = csv.writer(ndFile)
    #dWriter.writerow(header)
    #ndWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_COVAREP.csv'
        f = pd.read_csv(fileName, delimiter=',|\t')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100
            # for i in xrange(len(f)):
            #     if f.ix[i]['Frametime'] >= startTime and startFrame == 0:
            #         startFrame = i
            #     elif f.ix[i]['Frametime'] >= endTime and endFrame == 0:
            #         endFrame = i - 1

            features = f.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, item)
            vector.insert(1,instance[0])
            vector = np.asarray(vector)

            if questionType_DND[instance[0]] == 'D':
                dWriter.writerow(vector)
            else:
                ndWriter.writerow(vector)

def readCOVAREP_PN():
    groupByQuestion = {}
    pFile = open('data/pos_neg/positive_COVAREP.csv', 'a')
    nFile = open('data/pos_neg/negative_COVAREP.csv', 'a')
    pWriter = csv.writer(pFile)
    nWriter = csv.writer(nFile)
    #pWriter.writerow(header)
    #nWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_COVAREP.csv'
        f = pd.read_csv(fileName, delimiter=',|\t')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100
            # for i in xrange(len(f)):
            #     if f.ix[i]['Frametime'] >= startTime and startFrame == 0:
            #         startFrame = i
            #     elif f.ix[i]['Frametime'] >= endTime and endFrame == 0:
            #         endFrame = i - 1

            features = f.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1][:]
            vector += features
            vector.insert(0, item)
            vector.insert(1,instance[0])
            vector = np.asarray(vector)

            if questionType_PN[instance[0]] == 'P':
                pWriter.writerow(vector)
            else:
                nWriter.writerow(vector)

if __name__ == "__main__":
    readHelperData()
    readTranscript()
    #readFORMANT_DND()
    #readFORMANT_PN()
    readCOVAREP_DND()
    readCOVAREP_PN()