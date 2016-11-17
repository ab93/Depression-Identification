import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv
import os
import fnmatch

followUp = {}
ack = {}
nonIntimate = {}
intimate = {}
featureList = {}
questionType = {}

discriminativeVectors = []
nonDiscriminativeVectors = []


def readQuestions():
    global followUp, ack, nonIntimate, intimate
    utterrances = pd.read_csv('../data/IdentifyingFollowUps.csv')
    questions = pd.read_csv('../data/DND:Annotation-Supervised.csv')

    for i in range(len(questions)):
        question = questions.iloc[i]['Questions']
        qType = questions.iloc[i]['Annotations']
        questionType[question] = qType

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
    transcriptFiles = glob('../../../Data/[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')
    for i in range(0, len(transcriptFiles)):
        t = pd.read_csv(transcriptFiles[i], delimiter='\t')
        t = t.fillna("")
        captureStarted = False
        startTime = 0.0
        endTime = 0.0
        prevQuestion = ""
        participantNo=transcriptFiles[i][-18:-15]
        for j in range(len(t)):

            question = re.search(".*\((.*)\)$", t.iloc[j]['value'])
            if question is not None:
                question = question.group(1)
            else:
                question = t.iloc[j]['value']
            question = question.strip()

            if t.iloc[j]['speaker'] == 'Ellie':
                if question in nonIntimate and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in intimate and question in questionType and captureStarted:
                    # if '339' in transcriptFiles[i]:
                    #     print question
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    startTime = t.iloc[j]['start_time']
                    endTime = t.iloc[j]['stop_time']
                    prevQuestion = question

                elif question in intimate and question in questionType and not captureStarted:
                    startTime = t.iloc[j]['start_time']
                    endTime = t.iloc[j]['stop_time']
                    prevQuestion = question
                    captureStarted = True

                elif question in intimate and question not in questionType and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in followUp or question in ack and captureStarted:
                    endTime = t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker'] == 'Participant' and captureStarted:
                endTime = t.iloc[j]['stop_time']


def readCLM():
    groupByQuestion = {}

    dFile2 = open('../data/discriminative_CLM.csv', 'ab')
    ndFile2 = open('../data/nonDiscriminative_CLM.csv', 'ab')

    dWriter2 = csv.writer(dFile2)
    ndWriter2 = csv.writer(ndFile2)

    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:

        fileName2 = '../../../Data/' + item + '_P/' + item + '_CLM_features.csv'

        f2 = pd.read_csv(fileName2, delimiter=',')

        for instance in groupByQuestion[item]:

            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f2.ix[(f2[' timestamp'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f2.ix[(f2[' timestamp'] - endTime).abs().argsort()[:1]].index.tolist()[0]
            #print startFrame, endFrame

            features = f2.ix[startFrame:endFrame].mean(0).tolist()
            vector = instance[1]
            vector += features
            vector.insert(0, instance[0])
            vector.insert(0, item)
            vector = np.asarray(vector)
            #print(item, instance[0], instance[1][1], instance[1][2])

            if questionType[instance[0]] == 'D':
                dWriter2.writerow(vector)
            else:
                ndWriter2.writerow(vector)



def renameCLMFilesToCSV():

    rootdir = '../../Data/'

    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, '*CLM*.txt'):
            print(os.path.join(rootdir+'/'+filename))
            os.rename(os.path.join(root+'/'+filename), root+'/'+filename[:-4] + '.csv')


if __name__ == "__main__":
    readQuestions()
    readTranscript()
    renameCLMFilesToCSV()
    readCLM()