import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv

followUp = {}
ack = {}
nonIntimate = {}
intimate = {}
featureList={}
questionType={}

discriminativeVectors=[]
nonDiscriminativeVectors=[]

def readQuestions():
    global followUp, ack, nonIntimate, intimate
    utterrances = pd.read_csv('../Data/IdentifyingFollowUps.csv')
    questions=pd.read_excel('../data/QuestionsClassification.xlsx',sheetname='Annotation-Supervised')

    for i in xrange(len(questions)):
        question=questions.iloc[i]['Questions']
        qType=questions.iloc[i]['Annotations']
        questionType[question]=qType
        
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
    transcriptFiles=glob('../../Data/[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')
    for i in range(0,len(transcriptFiles)):
        t=pd.read_csv(transcriptFiles[i], delimiter='\t')
        captureStarted=False
        startTime=0.0
        endTime=0.0
        prevQuestion=""
        participantNo=transcriptFiles[i][11:14]
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

                elif question in intimate and question in questionType and captureStarted:
                    if (participantNo, prevQuestion) not in featureList:
                        featureList[(participantNo, prevQuestion)]=[startTime, endTime]
                    else:
                        featureList[(participantNo, prevQuestion)][1]=endTime
                    startTime=t.iloc[j]['start_time']
                    endTime=t.iloc[j]['stop_time']
                    prevQuestion=question

                elif question in intimate and question in questionType and not captureStarted:
                    startTime=t.iloc[j]['start_time']
                    endTime=t.iloc[j]['stop_time']
                    prevQuestion=question
                    captureStarted=True

                elif question in followUp or question in ack and captureStarted:
                    endTime=t.iloc[j]['stop_time']

            elif t.iloc[j]['speaker']=='Participant' and captureStarted:
                endTime=t.iloc[j]['stop_time']


def readFACET():
    facetFiles=glob('../../Data/[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_FACET_features.csv')
    groupByQuestion={}
    dFile=open('discriminativeVectors.csv','a')
    ndFile=open('nonDiscriminativeVectors.csv','a')
    dWriter=csv.writer(dFile)
    ndWriter=csv.writer(ndFile)
    for item in featureList:
        if item[0] not in groupByQuestion:
            groupByQuestion[item[0]]=[(item[1], featureList[item])]
        else:
            groupByQuestion[item[0]].append((item[1], featureList[item]))

    for item in groupByQuestion:
        fileName='../../Data/'+item+'_P/'+item+'_FACET_features.csv'
        f=pd.read_csv(fileName, delimiter=',')

        for instance in groupByQuestion[item]:
            startTime=instance[1][0]
            endTime=instance[1][1]

            startFrame=0
            endFrame=0
            for i in xrange(len(f)):
                if f.ix[i]['Frametime'] >= startTime and startFrame==0:
                    startFrame=i
                elif f.ix[i]['Frametime'] >= endTime and endFrame==0:
                    endFrame=i-1

            features=f.ix[startFrame:endFrame].mean(0).tolist()
            vector=instance[1]
            vector+=features
            vector.insert(0, item)
            vector=np.asarray(vector)
            print item, instance[0]
            if questionType[instance[0]]=='D':
                dWriter.writerow(vector)
            else:
                ndWriter.writerow(vector)




if __name__=="__main__":
    readQuestions()
    readTranscript()
    readFACET()