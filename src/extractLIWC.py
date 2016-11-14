import scipy.stats
import pandas as pd
from pprint import pprint
from glob import glob
import numpy as np
import re
import csv

# depressed=[]
# not_depressed=[]

# with open('data/liwc_depressed.csv') as f:
# 	reader=csv.reader(f)
# 	for row in reader:
# 		depressed.append(row[1:])

# with open('data/liwc_notdepressed.csv') as f:
# 	reader=csv.reader(f)
# 	for row in reader:
# 		not_depressed.append(row[1:])

# d_f=[[row[i] for row in depressed] for i in range(len(depressed[0]))]
# nd_f=[[row[i] for row in not_depressed] for i in range(len(not_depressed[0]))]

# for i in range(0,len(d_f)):
# 	for j in range(0,len(d_f[0])):
# 		d_f[i][j]=float(d_f[i][j])

# for i in range(0,len(nd_f)):
# 	for j in range(0,len(nd_f[0])):
# 		nd_f[i][j]=float(nd_f[i][j])

# features=[]
# for i in range(1,len(d_f)):
# 	t,p=scipy.stats.ttest_ind(d_f[i], nd_f[i], None, False)
# 	if p<=0.10:
# 		features.append(i)

followUp = {}
ack = {}
nonIntimate = {}
intimate = {}
featureList={}
questionType={}

discriminativeVectors=[]
nonDiscriminativeVectors=[]
questionAnswers={}
liwcVectors={}

def readUtterances():
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
        prevUtterance=""
        participantNo=transcriptFiles[i][11:14]
        responses=[]

        for j in xrange(len(t)):
            utterance=re.search(".*\((.*)\)$", t.iloc[j]['value'])
            if utterance is not None:
                utterance=utterance.group(1)
            else:
                utterance=t.iloc[j]['value']
            utterance=utterance.strip()

            if t.iloc[j]['speaker']=='Ellie':
                if utterance in nonIntimate and captureStarted:
                    if (participantNo, prevUtterance) not in featureList:
                        questionAnswers[(participantNo, prevUtterance)]=responses
                    else:
                        questionAnswers[(participantNo, prevUtterance)]+=responses

                    captureStarted=False
                    responses=[]

                elif utterance in intimate and utterance in questionType and captureStarted:
                    if (participantNo, prevUtterance) not in featureList:
                        questionAnswers[(participantNo, prevUtterance)]=responses
                    else:
                        questionAnswers[(participantNo, prevUtterance)]+=responses

                    prevUtterance=utterance
                    responses=[]

                elif utterance in intimate and utterance in questionType and not captureStarted:
                    prevUtterance=utterance
                    captureStarted=True

                elif utterance in followUp or utterance in ack and captureStarted:
                    continue

            elif t.iloc[j]['speaker']=='Participant' and captureStarted:
                responses.append(utterance)

def readLIWC():
    answerQuestion={}
    dFile=open('../data/discriminativeLIWC.csv','a')
    ndFile=open('../data/nonDiscriminativeLIWC.csv','a')
    dWriter=csv.writer(dFile)
    ndWriter=csv.writer(ndFile)
    for item in questionAnswers:
        for answer in questionAnswers[item]:
            if answer in answerQuestion:
                pass
            else:
                answerQuestion[answer]=(item[0], item[1])


    f=open('../Data/liwc.csv')
    reader=csv.reader(f)
    for row in reader:
        if row[0] not in liwcVectors:
            liwcVectors[row[0]]=[(row[1], row[2:])]
        else:
            liwcVectors[row[0]].append((row[1], row[2:]))

    #groupByVideo: (participantNo: [(question, answer)])
    #liwcVectors: (participantNo: [(answer, vector)])

    for video in liwcVectors:
        answerPair=liwcVectors[video]
        for item in answerPair:
            if item[0] in answerQuestion and questionType[answerQuestion[item[0]][1]]=='D':
                dWriter.writerow([video,answerQuestion[item[0]][1],item[0],item[1]])
            elif item[0] in answerQuestion and questionType[answerQuestion[item[0]][1]]=='ND':
                ndWriter.writerow([video,answerQuestion[item[0]][1],item[0],item[1]])


if __name__=="__main__":
    readUtterances()
    readTranscript()
    readLIWC()