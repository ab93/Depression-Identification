import pandas as pd
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

'''headers for COVAREP features'''

header = ["video", "question", "starttime", "endtime", 'F0_mean', 'VUV_mean', 'NAQ_mean', 'QOQ_mean', 'H1H2_mean',
          'PSP_mean', 'MDQ_mean', 'peakSlope_mean', 'Rd_mean', 'Rd_conf_mean', 'creak_mean', 'MCEP_0_mean',
          'MCEP_1_mean', 'MCEP_2_mean', 'MCEP_3_mean', 'MCEP_4_mean', 'MCEP_5_mean', 'MCEP_6_mean', 'MCEP_7_mean',
          'MCEP_8_mean', 'MCEP_9_mean', 'MCEP_10_mean', 'MCEP_11_mean', 'MCEP_12_mean', 'MCEP_13_mean', 'MCEP_14_mean',
          'MCEP_15_mean', 'MCEP_16_mean', 'MCEP_17_mean', 'MCEP_18_mean', 'MCEP_19_mean', 'MCEP_20_mean',
          'MCEP_21_mean', 'MCEP_22_mean', 'MCEP_23_mean', 'MCEP_24_mean', 'HMPDM_0_mean', 'HMPDM_1_mean',
          'HMPDM_2_mean', 'HMPDM_3_mean', 'HMPDM_4_mean', 'HMPDM_5_mean', 'HMPDM_6_mean', 'HMPDM_7_mean',
          'HMPDM_8_mean', 'HMPDM_9_mean', 'HMPDM_10_mean', 'HMPDM_11_mean', 'HMPDM_12_mean', 'HMPDM_13_mean',
          'HMPDM_14_mean', 'HMPDM_15_mean', 'HMPDM_16_mean', 'HMPDM_17_mean', 'HMPDM_18_mean', 'HMPDM_19_mean',
          'HMPDM_20_mean', 'HMPDM_21_mean', 'HMPDM_22_mean', 'HMPDM_23_mean', 'HMPDM_24_mean', 'HMPDD_0_mean',
          'HMPDD_1_mean', 'HMPDD_2_mean', 'HMPDD_3_mean', 'HMPDD_4_mean', 'HMPDD_5_mean', 'HMPDD_6_mean',
          'HMPDD_7_mean', 'HMPDD_8_mean', 'HMPDD_9_mean', 'HMPDD_10_mean', 'HMPDD_11_mean', 'HMPDD_12_mean',
          'F0_stddev', 'VUV_stddev', 'NAQ_stddev', 'QOQ_stddev', 'H1H2_stddev', 'PSP_stddev', 'MDQ_stddev',
          'peakSlope_stddev', 'Rd_stddev', 'Rd_conf_stddev', 'creak_stddev', 'MCEP_0_stddev', 'MCEP_1_stddev',
          'MCEP_2_stddev', 'MCEP_3_stddev', 'MCEP_4_stddev', 'MCEP_5_stddev', 'MCEP_6_stddev', 'MCEP_7_stddev',
          'MCEP_8_stddev', 'MCEP_9_stddev', 'MCEP_10_stddev', 'MCEP_11_stddev', 'MCEP_12_stddev', 'MCEP_13_stddev',
          'MCEP_14_stddev', 'MCEP_15_stddev', 'MCEP_16_stddev', 'MCEP_17_stddev', 'MCEP_18_stddev', 'MCEP_19_stddev',
          'MCEP_20_stddev', 'MCEP_21_stddev', 'MCEP_22_stddev', 'MCEP_23_stddev', 'MCEP_24_stddev', 'HMPDM_0_stddev',
          'HMPDM_1_stddev', 'HMPDM_2_stddev', 'HMPDM_3_stddev', 'HMPDM_4_stddev', 'HMPDM_5_stddev', 'HMPDM_6_stddev',
          'HMPDM_7_stddev', 'HMPDM_8_stddev', 'HMPDM_9_stddev', 'HMPDM_10_stddev', 'HMPDM_11_stddev', 'HMPDM_12_stddev',
          'HMPDM_13_stddev', 'HMPDM_14_stddev', 'HMPDM_15_stddev', 'HMPDM_16_stddev', 'HMPDM_17_stddev',
          'HMPDM_18_stddev', 'HMPDM_19_stddev', 'HMPDM_20_stddev', 'HMPDM_21_stddev', 'HMPDM_22_stddev',
          'HMPDM_23_stddev', 'HMPDM_24_stddev', 'HMPDD_0_stddev', 'HMPDD_1_stddev', 'HMPDD_2_stddev', 'HMPDD_3_stddev',
          'HMPDD_4_stddev', 'HMPDD_5_stddev', 'HMPDD_6_stddev', 'HMPDD_7_stddev', 'HMPDD_8_stddev', 'HMPDD_9_stddev',
          'HMPDD_10_stddev', 'HMPDD_11_stddev', 'HMPDD_12_stddev']

'''headers for FORMANT features'''

header_f = ["video", "question", "starttime", "endtime", 'formant1_mean', 'formant2_mean', 'formant3_mean',
            'formant4_mean', 'formant5_mean', 'formant1_stddev', 'formant2_stddev', 'formant3_stddev',
            'formant4_stddev', 'formant5_stddev']

questionType_DND = {}
questionType_PN = {}
questionAnswers = {}

'''
Reads DND questions and PN questions.
Retrieves acknowledgements, follow ups, intimate and non intimate questions and stores in global variables
'''


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


'''
Reads transcripts, captures the start and end times of the answers for most frequent intimate questions. Also captures the start and end times of follow up questions that are following most frequent intimate questions
'''


def readTranscript():
    global featureList
    transcriptFiles = glob(sys.argv[1] + '[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')

    for i in range(0, len(transcriptFiles)):
        t = pd.read_csv(transcriptFiles[i], delimiter=',|\t', engine='python')
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
                endTime = t.iloc[j]['start_time']
                if question in nonIntimate and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in intimate and question in questionType_DND and captureStarted:
                    endTime = t.iloc[j]['start_time']
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
                    endTime = t.iloc[j]['start_time']
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)] = [startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1] = endTime
                    captureStarted = False

                elif question in followUp or question in ack and captureStarted:
                    endTime = t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker'] == 'Participant' and captureStarted:
                # endTime=t.iloc[j]['stop_time']
                continue


'''
Generates features from FORMANT files considering the start and end times for each frequent intimate questions from PN list.
Features are generated by taking mean and std dev of all the features for every question for every video
'''


def readFORMANT_DND():
    print 'FORMANT DND'
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
        f = pd.read_csv(fileName, delimiter=',|\t', engine='python')

        for instance in groupByQuestion[item]:
            print item
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100

            features_mean = f.ix[startFrame:endFrame].mean(0).tolist()
            features_stddev = f.ix[startFrame:endFrame].std(0).tolist()

            vector = instance[1][:]
            vector += features_mean
            vector += features_stddev
            vector.insert(0, item)
            vector.insert(1, instance[0])
            vector = np.asarray(vector)
            if questionType_DND[instance[0]] == 'D':
                dWriter.writerow(vector)
            else:
                ndWriter.writerow(vector)


'''
Generates features from FORMANT files considering the start and end times for each frequent intimate questions from PN list.
Features are generated by taking mean and std dev of all the features for every question for every video
'''


def readFORMANT_PN():
    print 'FORMANT PN'
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
        f = pd.read_csv(fileName, delimiter=',|\t', engine='python')

        for instance in groupByQuestion[item]:
            print item
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100

            features_mean = f.ix[startFrame:endFrame].mean(0).tolist()
            features_stddev = f.ix[startFrame:endFrame].std(0).tolist()
            vector += features_mean
            vector += features_stddev

            vector.insert(0, item)
            vector.insert(1, instance[0])
            vector = np.asarray(vector)
            if questionType_PN[instance[0]] == 'P':
                pWriter.writerow(vector)
            else:
                nWriter.writerow(vector)


'''
Generates features from COVAREP files considering the start and end times for each frequent intimate questions from DND list.
Features are generated by taking mean and std dev of all the features for every question for every video
'''


def readCOVAREP_DND():
    print 'COVAREP DND'
    groupByQuestion = {}
    dFile = open('data/disc_nondisc/discriminative_COVAREP.csv', 'w')
    ndFile = open('data/disc_nondisc/nondiscriminative_COVAREP.csv', 'w')
    dWriter = csv.writer(dFile)
    ndWriter = csv.writer(ndFile)
    dWriter.writerow(header)
    ndWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_COVAREP.csv'
        f = pd.read_csv(fileName, delimiter=',|\t', engine='python')

        for instance in groupByQuestion[item]:
            print item
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100

            features_mean = f.ix[startFrame:endFrame].mean(0).tolist()
            features_stddev = f.ix[startFrame:endFrame].std(0).tolist()

            vector = instance[1][:]
            vector += features_mean
            vector += features_stddev
            vector.insert(0, item)
            vector.insert(1, instance[0])
            vector = np.asarray(vector)

            if questionType_DND[instance[0]] == 'D':
                dWriter.writerow(vector)
            else:
                ndWriter.writerow(vector)


'''
Generates features from COVAREP files considering the start and end times for each frequent intimate questions from PN list.
Features are generated by taking mean and std dev of all the features for every question for every video
'''


def readCOVAREP_PN():
    print 'COVAREP PN'
    groupByQuestion = {}
    pFile = open('data/pos_neg/positive_COVAREP.csv', 'w')
    nFile = open('data/pos_neg/negative_COVAREP.csv', 'w')
    pWriter = csv.writer(pFile)
    nWriter = csv.writer(nFile)
    pWriter.writerow(header)
    nWriter.writerow(header)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]] = [(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName = sys.argv[1] + item + '_P/' + item + '_COVAREP.csv'
        f = pd.read_csv(fileName, delimiter=',|\t', engine='python')

        for instance in groupByQuestion[item]:
            print item
            startTime = instance[1][0]
            endTime = instance[1][1]

            startFrame = startTime * 100
            endFrame = endTime * 100

            features_mean = f.ix[startFrame:endFrame].mean(0).tolist()
            features_stddev = f.ix[startFrame:endFrame].std(0).tolist()
            vector = instance[1][:]
            vector += features_mean
            vector += features_stddev
            vector.insert(0, item)
            vector.insert(1, instance[0])
            vector = np.asarray(vector)

            if questionType_PN[instance[0]] == 'P':
                pWriter.writerow(vector)
            else:
                nWriter.writerow(vector)


if __name__ == "__main__":
    readHelperData()
    readTranscript()
    readFORMANT_DND()
    readFORMANT_PN()
    readCOVAREP_DND()
    readCOVAREP_PN()
