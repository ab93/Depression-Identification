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
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in intimate and question in questionType_DND and captureStarted:
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
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in followUp or question in ack and captureStarted:
                    endTime = t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker'] == 'Participant' and captureStarted:
                endTime = t.iloc[j]['stop_time']


def readCLM3D_DND():
    groupByQuestion = {}

    dFile = open('data/disc_nondisc/discriminative_CLM_3D.csv', 'w')
    ndFile = open('data/disc_nondisc/nondiscriminative_CLM_3D.csv', 'w')

    dWriter = csv.writer(dFile)
    ndWriter = csv.writer(ndFile)

    header = ["frame", "timestamp", "confidence", "success", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9",
              "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24",
              "X25", "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33", "X34", "X35", "X36", "X37", "X38", "X39",
              "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49", "X50", "X51", "X52", "X53", "X54",
              "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64", "X65", "X66", "X67", "Y0", "Y1", "Y2",
              "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10", "Y11", "Y12", "Y13", "Y14", "Y15", "Y16", "Y17", "Y18",
              "Y19", "Y20", "Y21", "Y22", "Y23", "Y24", "Y25", "Y26", "Y27", "Y28", "Y29", "Y30", "Y31", "Y32", "Y33",
              "Y34", "Y35", "Y36", "Y37", "Y38", "Y39", "Y40", "Y41", "Y42", "Y43", "Y44", "Y45", "Y46", "Y47", "Y48",
              "Y49", "Y50", "Y51", "Y52", "Y53", "Y54", "Y55", "Y56", "Y57", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63",
              "Y64", "Y65", "Y66", "Y67", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12",
              "Z13", "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", "Z22", "Z23", "Z24", "Z25", "Z26", "Z27", "Z28",
              "Z29", "Z30", "Z31", "Z32", "Z33", "Z34", "Z35", "Z36", "Z37", "Z38", "Z39", "Z40", "Z41", "Z42", "Z43", "Z44",
              "Z45", "Z46", "Z47", "Z48", "Z49", "Z50", "Z51", "Z52", "Z53", "Z54", "Z55", "Z56", "Z57", "Z58", "Z59",
              "Z60", "Z61", "Z62", "Z63", "Z64", "Z65", "Z66", "Z67"]



    dWriter.writerow(header)
    ndWriter.writerow(header)

    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_CLM_features3D.csv'
        f = pd.read_csv(fileName, delimiter=',')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f.ix[(f['Frametime'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f.ix[(f['Frametime'] - endTime).abs().argsort()[:1]].index.tolist()[0]
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


def readCLM3D_PN():

    groupByQuestion = {}
    pFile = open('data/pos_neg/positive_CLM_3D.csv', 'w')
    nFile = open('data/pos_neg/negative_CLM_3D.csv', 'w')
    pWriter = csv.writer(pFile)
    nWriter = csv.writer(nFile)
    header = ["frame", "timestamp", "confidence", "success", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9",
              "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24",
              "X25", "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33", "X34", "X35", "X36", "X37", "X38", "X39",
              "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49", "X50", "X51", "X52", "X53", "X54",
              "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64", "X65", "X66", "X67", "Y0", "Y1", "Y2",
              "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10", "Y11", "Y12", "Y13", "Y14", "Y15", "Y16", "Y17", "Y18",
              "Y19", "Y20", "Y21", "Y22", "Y23", "Y24", "Y25", "Y26", "Y27", "Y28", "Y29", "Y30", "Y31", "Y32", "Y33",
              "Y34", "Y35", "Y36", "Y37", "Y38", "Y39", "Y40", "Y41", "Y42", "Y43", "Y44", "Y45", "Y46", "Y47", "Y48",
              "Y49", "Y50", "Y51", "Y52", "Y53", "Y54", "Y55", "Y56", "Y57", "Y58", "Y59", "Y60", "Y61", "Y62", "Y63",
              "Y64", "Y65", "Y66", "Y67", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12",
              "Z13", "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", "Z22", "Z23", "Z24", "Z25", "Z26", "Z27", "Z28",
              "Z29", "Z30", "Z31", "Z32", "Z33", "Z34", "Z35", "Z36", "Z37", "Z38", "Z39", "Z40", "Z41", "Z42", "Z43", "Z44",
              "Z45", "Z46", "Z47", "Z48", "Z49", "Z50", "Z51", "Z52", "Z53", "Z54", "Z55", "Z56", "Z57", "Z58", "Z59",
              "Z60", "Z61", "Z62", "Z63", "Z64", "Z65", "Z66", "Z67"]

    pWriter.writerow(header)
    nWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_CLM_features3D.csv'
        f = pd.read_csv(fileName, delimiter=',')

        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = f.ix[(f['Frametime'] - startTime).abs().argsort()[:1]].index.tolist()[0]
            endFrame = f.ix[(f['Frametime'] - endTime).abs().argsort()[:1]].index.tolist()[0]

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
    readCLM3D_DND()
    readCLM3D_PN()