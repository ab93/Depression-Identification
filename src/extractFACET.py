import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re

followUp = {}
ack = {}
nonIntimate = {}
intimate = {}

featureList={}
def readQuestions():
    global followUp, ack, nonIntimate, intimate
    questions = pd.read_csv('../Data/IdentifyingFollowUps.csv')
    for item in questions.itertuples():
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
        participantNo=transcriptFiles[i][11:14]
        featureList[participantNo]=[]
        for i in xrange(len(t)):
            # print t.iloc[i]
            question=re.search(".*\((.*)\)$", t.iloc[i]['value'])
            if question is not None:
                question=question.group(1)
            else:
                question=t.iloc[i]['value']

            if t.iloc[i]['speaker']=='Ellie':
                if question in nonIntimate and captureStarted:
                    featureList[participantNo].append([startTime, endTime])
                    captureStarted=False

                elif question in intimate and captureStarted:
                    featureList[participantNo].append([startTime, endTime])
                    startTime=t.iloc[i]['start_time']
                    endTime=t.iloc[i]['stop_time']

                elif question in intimate and not captureStarted:
                    startTime=t.iloc[i]['start_time']
                    endTime=t.iloc[i]['stop_time']
                    captureStarted=True

                elif question in followUp or question in ack:
                    endTime=t.iloc[i]['stop_time']

            elif t.iloc[i]['speaker']=='Participant' and captureStarted:
                endTime=t.iloc[i]['stop_time']

def readFACET():
    facetFiles=glob('../../Data/[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_FACET_features.csv')

    for item in featureList:
        fileName='../../Data/'+item+'_P/'+item+'_FACET_features.csv'
        f=pd.read_csv(fileName, delimiter=',')
        for instance in featureList[item]:
            startTime=instance[0]
            endTime=instance[1]

            startFrame=0
            endFrame=0
            for i in xrange(len(f)):
                if f.iloc[i]['Frametime'] >= startTime and startFrame==0:
                    startFrame=i
                elif f.iloc[i]['Frametime'] >= endTime and endFrame==0:
                    endFrame=i-1
            features=f.iloc[[startFrame, endFrame]].mean(0).tolist()
            instance+=features
            instance.insert(0, item)
            instance=np.asarray(instance)
            print instance




if __name__=="__main__":
    readQuestions()
    readTranscript()
    readFACET()