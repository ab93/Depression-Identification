import scipy.stats
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
featureList={}
questionType_DND={}
questionType_PN={}

discriminativeVectors=[]
nonDiscriminativeVectors=[]
questionAnswers={}
liwcVectors={}

listofParticipants=[]
def readHelperData():
    global followUp, ack, nonIntimate, intimate, questionType_PN, questionType_DND
    utterrances = pd.read_csv('data/misc/IdentifyingFollowUps.csv')
    disc_nondisc = pd.read_csv('data/misc/DND_Annotations.csv')
    pos_neg = pd.read_csv('data/misc/PN_Annotations.csv')

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
        t=pd.read_csv(transcriptFiles[i], delimiter='\t')
        t = t.fillna("")
        captureStarted=False
        prevUtterance=""
        participantNo=transcriptFiles[i][-18:-15]

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

                elif utterance in intimate and utterance in questionType_DND and captureStarted:
                    if (participantNo, prevUtterance) not in featureList:
                        questionAnswers[(participantNo, prevUtterance)]=responses
                    else:
                        questionAnswers[(participantNo, prevUtterance)]+=responses

                    prevUtterance=utterance
                    responses=[]

                elif utterance in intimate and utterance in questionType_DND and not captureStarted:
                    prevUtterance=utterance
                    captureStarted=True

                elif utterance in intimate and utterance not in questionType_DND and captureStarted:
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

def readLIWC_DND():
    global listofParticipants
    answerQuestion={}
    dFile=open('data/disc_nondisc/discriminative_LIWC.csv','w')
    ndFile=open('data/disc_nondisc/nondiscriminative_LIWC.csv','w')
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


    f=open('data/misc/liwc.csv')
    reader=csv.reader(f)
    header=['video','question']
    header+=reader.next()[2:]
    dWriter.writerow(header)
    ndWriter.writerow(header)
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
            if item[0] in answerQuestion and questionType_DND[answerQuestion[item[0]][1]]=='D':
                vector=[float(i) for i in item[1]]
                vector.insert(0,answerQuestion[item[0]][1])
                vector.insert(0,str(video))
                discriminativeMatrix.append(vector)


            elif item[0] in answerQuestion and questionType_DND[answerQuestion[item[0]][1]]=='ND':
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

def readLIWC_PN():
    global listofParticipants
    answerQuestion={}
    pFile=open('data/pos_neg/positive_LIWC.csv','w')
    nFile=open('data/pos_neg/negative_LIWC.csv','w')
    pWriter=csv.writer(pFile)
    nWriter=csv.writer(nFile)
    positiveDF=pd.DataFrame()
    negativeDF=pd.DataFrame()

    positiveMatrix=[]
    negativeMatrix=[]
    for item in questionAnswers:
        for answer in questionAnswers[item]:
            if answer in answerQuestion:
                pass
            else:
                answerQuestion[answer]=(item[0], item[1])


    f=open('data/misc/liwc.csv')
    reader=csv.reader(f)
    header=['video','question']
    header+=reader.next()[2:]
    pWriter.writerow(header)
    nWriter.writerow(header)
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
            if item[0] in answerQuestion and questionType_PN[answerQuestion[item[0]][1]]=='P':
                vector=[float(i) for i in item[1]]
                vector.insert(0,answerQuestion[item[0]][1])
                vector.insert(0,str(video))
                positiveMatrix.append(vector)


            elif item[0] in answerQuestion and questionType_PN[answerQuestion[item[0]][1]]=='N':
                vector=[float(i) for i in item[1]]
                vector.insert(0,answerQuestion[item[0]][1])
                vector.insert(0,str(video))
                negativeMatrix.append(vector)

    positiveDF=pd.DataFrame(positiveMatrix, columns=header)
    negativeDF=pd.DataFrame(negativeMatrix, columns=header)
    for k1, k2 in positiveDF.groupby(['video','question']):
        vec=[k1[0],k1[1]]
        x=k2.mean().values.tolist()
        vec+=x
        pWriter.writerow(vec)

    for k1, k2 in negativeDF.groupby(['video','question']):
        vec=[k1[0],k1[1]]
        x=k2.mean().values.tolist()
        vec+=x
        nWriter.writerow(vec)


if __name__=="__main__":
    readHelperData()
    readTranscript()
    readLIWC_DND()
    readLIWC_PN()