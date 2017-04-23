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

'''
Reads the following data:
IdentifyingFollowUps.csv : Reads tags for each question in the corpus where the tag is intimate, non-intimate,
                            acknowledgement, follow-up
DND_Annotations.csv : Reads category for each question in the corpus where the category is Discriminative, Non-discriminative

PN_Annotations.csv : Reads category for each question in the corpus where the category is Positive, Negative
'''

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

'''
Reads the transcript of each interview and collects the answer for each question asked to a participant
This is done for all the interviews.
At the end, the dictionary questionAnswers has a list of utterances by the participant (the utterances together make up the answer)
and this is stored as [(300,'when was the last time you felt happy)] = ['i last felt happy','um','yesterday]
In this manner, responses are collected for all questions, for all interviews
'''
def readTranscript():
    global featureList
    transcriptFiles=glob(sys.argv[1]+'[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')
    for i in range(0,len(transcriptFiles)):
        t=pd.read_csv(transcriptFiles[i], delimiter=',|\t')
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

            '''
            We start capturing a response right after Ellie asks an intimate question.
            If a question is a follow-up question, we continue capturing participant response after Ellie
            asks the follow-up question.

            If Ellie gives an acknowledgement, we skip and go on. If we are capturing a response already, we continue
            capturing from the next participant utterance.

            If a question is a non-intimate question, then there are two cases:
            1. Previous question was an intimate question: In this case, we have been capturing participant
            responses. So, we stop capture and store it in the dictionary. We skip the non-intimate question
            and wait until next time Ellie asks an intimate question or conversation ends, whichever is first.
            2. Previous question was a non-intimate question: In this case, we skip this question and wait till
            Ellie asks an intimate question or conversation ends, whichever is first.
            '''
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

'''
Reads LIWC features for all questions that are either Discriminative or Non-discriminative
and writes it to file.
'''
def readLIWC_DND():
    global listofParticipants
    answerQuestion={}
    dFile=open('data/disc_nondisc/discriminative_LIWC.csv','w')
    ndFile=open('data/disc_nondisc/nondiscriminative_LIWC.csv','w')
    dWriter=csv.writer(dFile)
    ndWriter=csv.writer(ndFile)

    f=open('data/misc/liwc_new.csv')
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

    #questionAnswers: [(participantNo, question)]=[list of answers]
    #liwcVectors: participantNo: [list of (answer, vector)]
    '''
    A participant response for a question will span multiple rows in the transcript.
    Now, we obtain the LIWC feature vector for each of these responses from the liwc_new.csv file.
    To get the full LIWC vector for the entire response (set of utterance - think of the response as a single paragraph),
    each value in the feature vector is multiplied by the utterance_length (number of words in one utterances).
    Finally, for each LIWC feature, we sum the values of that feature for all utterances and divide by total 
    number of words in the response (average over number of words).

    For example,
    i was not happy...     0.40    0.62    0.33    ...
    yesterday...           0.3     0.00    0.20    ...
    but...                 0.00    0.10    0.80    ...

    Multiplied vectors: (line 194)
    i was not happy...     1.60    2.48    1.32    ... (Multiplied by 4 - number of words)
    yesterday...           0.3     0.00    0.20    ... (Multiplied by 1)
    but no...              0.00    0.20    1.60    ... (Multiplied by 2)

    Averaged vectors: (denominator=7 which is total number of words) (line 196)
    i was not happy yesterday but no...0.27    0.38    0.44

    This averaged vector is written to file. This vector is the LIWC vector for the entire response of the 
    participant for that question. This is done in readLIWC_PN also, for the positive and negative category
    questions.
    '''
    for item in questionAnswers:
        participant_number=item[0]
        current_question=item[1]
        lines_for_this_question=[]
        answer_length=0.0
        for answer in questionAnswers[item]:
            vectors=liwcVectors[participant_number]
            for vector in vectors:
                if answer==vector[0]:
                    utterance_length=len(answer.split(" "))
                    answer_length+=utterance_length
                    feature_vector=[float(i)*utterance_length for i in vector[1]]
                    lines_for_this_question.append(feature_vector)
        final_vector=[sum(value)/answer_length for value in zip(*lines_for_this_question)]
        final_vector.insert(0,current_question)
        final_vector.insert(0,str(participant_number))

        if questionType_DND[current_question]=='D':
            dWriter.writerow(final_vector)
        elif questionType_DND[current_question]=='ND':
            ndWriter.writerow(final_vector)

'''
Reads LIWC features for all questions that are either Positive or Negative
and writes it to file.
'''
def readLIWC_PN():
    global listofParticipants
    answerQuestion={}
    pFile=open('data/pos_neg/positive_LIWC.csv','w')
    nFile=open('data/pos_neg/negative_LIWC.csv','w')
    pWriter=csv.writer(pFile)
    nWriter=csv.writer(nFile)

    f=open('data/misc/liwc_new.csv')
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

    #questionAnswers: [(participantNo, question)]=[list of answers]
    #liwcVectors: participantNo: [list of (answer, vector)]

    for item in questionAnswers:
        participant_number=item[0]
        current_question=item[1]
        lines_for_this_question=[]
        answer_length=0.0
        for answer in questionAnswers[item]:
            vectors=liwcVectors[participant_number]
            for vector in vectors:
                if answer==vector[0]:
                    utterance_length=len(answer.split(" "))
                    answer_length+=utterance_length
                    feature_vector=[float(i)*utterance_length for i in vector[1]]
                    lines_for_this_question.append(feature_vector)
        final_vector=[sum(value)/answer_length for value in zip(*lines_for_this_question)]
        final_vector.insert(0,current_question)
        final_vector.insert(0,str(participant_number))

        if questionType_PN[current_question]=='P':
            pWriter.writerow(final_vector)
        elif questionType_PN[current_question]=='N':
            nWriter.writerow(final_vector)

if __name__=="__main__":
    readHelperData()
    readTranscript()
    readLIWC_DND()
    readLIWC_PN()