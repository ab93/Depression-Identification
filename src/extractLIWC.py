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

listofParticipants=[]
def readUtterances():
    global followUp, ack, nonIntimate, intimate
    utterrances = pd.read_csv('../Data/IdentifyingFollowUps.csv')
    questions=pd.read_csv('../data/DND:Annotation-Supervised.csv')

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
        t = t.fillna("")
        captureStarted=False
        prevUtterance=""
        participantNo=transcriptFiles[i][11:14]
        listofParticipants.append(participantNo)
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

                elif utterance in intimate and utterance not in questionType and captureStarted:
                    if (participantNo, prevUtterance) not in featureList:
                        questionAnswers[(participantNo, prevUtterance)]=responses
                    else:
                        questionAnswers[(participantNo, prevUtterance)]+=responses

                    captureStarted=False
                    responses=[]

                elif utterance in followUp or utterance in ack and captureStarted:
                    continue

            elif t.iloc[j]['speaker']=='Participant' and captureStarted:
                responses.append(utterance)

def readLIWC():
    global listofParticipants
    answerQuestion={}
    dFile=open('../data/discriminativeLIWC.csv','a')
    ndFile=open('../data/nonDiscriminativeLIWC.csv','a')
    dWriter=csv.writer(dFile)
    ndWriter=csv.writer(ndFile)
    discriminativeDF=pd.DataFrame()
    nonDiscriminativeDF=pd.DataFrame()

    discriminativeMatrix=[]
    nonDiscriminativeMatrix=[]
    for item in questionAnswers:
        for answer in questionAnswers[item]:
            if answer in answerQuestion:
                pass
            else:
                answerQuestion[answer]=(item[0], item[1])


    f=open('../Data/liwc.csv')
    reader=csv.reader(f)
    header=['video','question']
    header+=reader.next()[2:]

    listofParticipants=[int(i) for i in listofParticipants]

    listofParticipants.sort()
    
    for row in reader:
        if int(row[0])>=listofParticipants[0] and int(row[0])<=listofParticipants[-1]:
            if row[0] not in liwcVectors:
                liwcVectors[row[0]]=[(row[1], row[2:])]
            else:
                liwcVectors[row[0]].append((row[1], row[2:]))

    #answerQuestion: answer: [participantNo, question]
    #liwcVectors: participantNo: [(answer, vector)])

    for video in liwcVectors:
        answerPair=liwcVectors[video]

        for item in answerPair:
            if item[0] in answerQuestion and questionType[answerQuestion[item[0]][1]]=='D' :
                vector=[float(i) for i in item[1]]
                vector.insert(0,answerQuestion[item[0]][1])
                vector.insert(0,str(video))
                discriminativeMatrix.append(vector)


            elif item[0] in answerQuestion and questionType[answerQuestion[item[0]][1]]=='ND':
                vector=[float(i) for i in item[1]]
                vector.insert(0,answerQuestion[item[0]][1])
                vector.insert(0,str(video))
                nonDiscriminativeMatrix.append(vector)

    discriminativeDF=pd.DataFrame(discriminativeMatrix, columns=header)
    nonDiscriminativeDF=pd.DataFrame(nonDiscriminativeMatrix, columns=header)
    for k1, k2 in discriminativeDF.groupby(['video','question']):
        vec=[k1[0],k1[1]]
        x=k2.mean().values.tolist()
        vec+=x
        dWriter.writerow(vec)

    for k1, k2 in nonDiscriminativeDF.groupby(['video','question']):
        vec=[k1[0],k1[1]]
        x=k2.mean().values.tolist()
        vec+=x
        ndWriter.writerow(vec)

    # discriminativeDF=discriminativeDF.groupby(['video','question']).mean()
    # nonDiscriminativeDF=nonDiscriminativeDF.groupby(['video','question']).mean()
    # #pprint(discriminativeDF.iloc[0])
    
    # discriminativeDF=discriminativeDF.values.tolist()
    # nonDiscriminativeDF=nonDiscriminativeDF.values.tolist()


    # for vec in discriminativeDF:
    #     dWriter.writerow(vec)

    # for vec in nonDiscriminativeDF:
    #     ndWriter.writerow(vec)


if __name__=="__main__":
    readUtterances()
    readTranscript()
    readLIWC()