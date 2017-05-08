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
question_number_mapping = {}


discriminativeVectors = []
nonDiscriminativeVectors = []
'''headers for OPENFACE features'''
header = ["video", "question", "question_number","starttime", "endtime", 'frame_mean', 'timestamp_mean', 'confidence_mean',
          'success_mean', 'pose_Tx_mean', 'pose_Ty_mean', 'pose_Tz_mean', 'pose_Rx_mean', 'pose_Ry_mean',
          'pose_Rz_mean', 'AU01_r_mean', 'AU02_r_mean', 'AU04_r_mean', 'AU05_r_mean', 'AU06_r_mean', 'AU07_r_mean',
          'AU09_r_mean', 'AU10_r_mean', 'AU12_r_mean', 'AU14_r_mean', 'AU15_r_mean', 'AU17_r_mean', 'AU20_r_mean',
          'AU23_r_mean', 'AU25_r_mean', 'AU26_r_mean', 'AU45_r_mean', 'AU01_c_mean', 'AU02_c_mean', 'AU04_c_mean',
          'AU05_c_mean', 'AU06_c_mean', 'AU07_c_mean', 'AU09_c_mean', 'AU10_c_mean', 'AU12_c_mean', 'AU14_c_mean',
          'AU15_c_mean', 'AU17_c_mean', 'AU20_c_mean', 'AU23_c_mean', 'AU25_c_mean', 'AU26_c_mean', 'AU28_c_mean',
          'AU45_c_mean', 'frame_stddev', 'timestamp_stddev', 'confidence_stddev', 'success_stddev', 'pose_Tx_stddev',
          'pose_Ty_stddev', 'pose_Tz_stddev', 'pose_Rx_stddev', 'pose_Ry_stddev', 'pose_Rz_stddev', 'AU01_r_stddev',
          'AU02_r_stddev', 'AU04_r_stddev', 'AU05_r_stddev', 'AU06_r_stddev', 'AU07_r_stddev', 'AU09_r_stddev',
          'AU10_r_stddev', 'AU12_r_stddev', 'AU14_r_stddev', 'AU15_r_stddev', 'AU17_r_stddev', 'AU20_r_stddev',
          'AU23_r_stddev', 'AU25_r_stddev', 'AU26_r_stddev', 'AU45_r_stddev', 'AU01_c_stddev', 'AU02_c_stddev',
          'AU04_c_stddev', 'AU05_c_stddev', 'AU06_c_stddev', 'AU07_c_stddev', 'AU09_c_stddev', 'AU10_c_stddev',
          'AU12_c_stddev', 'AU14_c_stddev', 'AU15_c_stddev', 'AU17_c_stddev', 'AU20_c_stddev', 'AU23_c_stddev',
          'AU25_c_stddev', 'AU26_c_stddev', 'AU28_c_stddev', 'AU45_c_stddev','gender']

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
    #create question string to number mapping
    count = 0
    for question in questionType_DND:
        question_number_mapping[question] = count
        count += 1

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
                if question in nonIntimate and captureStarted:
                    endTime = t.iloc[j]['start_time']
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
                continue
                # endTime = t.iloc[j]['stop_time']


'''
Generates features from OPENFACE files considering the start and end times for each frequent intimate questions from DND list.
Features are generated by taking mean and std dev of all the features for every question for every video
'''


def readOPENFACE_DND(split_seconds):
    print "DND"
    groupByQuestion = {}
    gender=pd.Series.from_csv('data/misc/gender.csv').to_dict()
    dFile = open('data/disc_nondisc/discriminative_OPENFACE.csv', 'w')
    ndFile = open('data/disc_nondisc/nondiscriminative_OPENFACE.csv', 'w')
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
        fileName = sys.argv[1] + item + '_P/' + item + '_OPENFACE.txt'
        f = pd.read_csv(fileName, delimiter=', ', engine='python')
        print item
        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]

            time_split = startTime
            while time_split < endTime:
                startFrame = f.ix[(f['timestamp'] - time_split).abs().argsort()[:1]].index.tolist()[0]
                endFrame = f.ix[(f['timestamp'] - min((time_split+split_seconds),endTime)).abs().argsort()[:1]].index.tolist()[0]

                #endFrame = f.ix[(f['timestamp'] - min(time_split+split_seconds,endTime)).abs().argsort()[:1]].index.tolist()[0]
                features_mean = f.ix[startFrame:endFrame].mean(0).tolist()
                features_stddev = f.ix[startFrame:endFrame].std(0).tolist()
                if len(features_mean)>45:
                    features_mean = features_mean[:45]
                if len(features_stddev)>45:
                    features_stddev = features_stddev[:45]
                vector = instance[1][:]
                vector += features_mean
                vector += features_stddev
                vector.insert(0, item)
                vector.insert(1, instance[0])                
                vector.insert(2, question_number_mapping[instance[0]])
                vector.append(gender[item])
                vector = np.asarray(vector)

                if questionType_DND[instance[0]] == 'D':
                    dWriter.writerow(vector)
                else:
                    ndWriter.writerow(vector)
                time_split += split_seconds
    dFile.close()
    ndFile.close()


'''
Generates features from OPENFACE files considering the start and end times for each frequent intimate questions from PN list.
Features are generated by taking mean and std dev of all the features for every question for every video
'''


def readOPENFACE_PN(split_seconds):
    print "PN"
    groupByQuestion = {}
    gender=pd.Series.from_csv('data/misc/gender.csv').to_dict()
    pFile = open('data/pos_neg/positive_OPENFACE.csv', 'w')
    nFile = open('data/pos_neg/negative_OPENFACE.csv', 'w')
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
        fileName = sys.argv[1] + item + '_P/' + item + '_OPENFACE.txt'
        f = pd.read_csv(fileName, delimiter=', ', engine='python')
        print item
        for instance in groupByQuestion[item]:
            startTime = instance[1][0]
            endTime = instance[1][1]
            
            time_split = startTime
            while time_split < endTime:
                startFrame = f.ix[(f['timestamp'] - time_split).abs().argsort()[:1]].index.tolist()[0]
                #endFrame = f.ix[(f['timestamp'] - min(time_split+split_seconds,endTime).abs().argsort()[:1]].index.tolist()[0]
                endFrame = f.ix[(f['timestamp'] - min((time_split + split_seconds),endTime)).abs().argsort()[:1]].index.tolist()[0]

                features_mean = f.ix[startFrame:endFrame].mean(0).tolist()
                features_stddev = f.ix[startFrame:endFrame].std(0).tolist()
                if len(features_mean)>45:
                    features_mean = features_mean[:45]
                if len(features_stddev)>45:
                    features_stddev = features_stddev[:45]
                vector = instance[1][:]
                vector += features_mean
                vector += features_stddev
                vector.insert(0, item)
                vector.insert(1, instance[0])
                vector.insert(2, question_number_mapping[instance[0]])
                vector.append(gender[item])
                vector = np.asarray(vector)
                # print item, instance[0], startTime, endTime

                if questionType_PN[instance[0]] == 'P':
                    pWriter.writerow(vector)
                else:
                    nWriter.writerow(vector)
                time_split += split_seconds
    pFile.close()
    nFile.close()


if __name__ == "__main__":
    try:
        split_seconds = int(sys.argv[2])
    except:
        print "Give folder name and split seconds"
    readHelperData()
    readTranscript()
    readOPENFACE_DND(split_seconds)
    readOPENFACE_PN(split_seconds)
