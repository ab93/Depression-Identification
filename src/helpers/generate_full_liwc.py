#Michelle Morales
#Extract LIWC features using 2015 English version

from __future__ import division
from collections import defaultdict
import re
import os
from glob import glob
import sys
import pandas as pd
import csv
features=[]
category_names=[]
""" Get features using LIWC 2015. categories in total."""
categoryIDs = {} #keep track of each category number
liwcD = {} #create liwc dictionary where each liwc dictionary word is a key that maps to a list that contains the liwc categories for that word
liwc_file = 'data/misc/LIWC2015_English.dic'#path to LIWC dict
read = open(liwc_file,'r').readlines()
header = read[1:77] #change this number depending on how many liwc categories you want to use
for line in header:
    items = line.strip().split()
    number,category_name = items[0],items[1]
    categoryIDs[number]=category_name
liwc_words = read[88:]#liwc dictionary words

for line in liwc_words:
    items = line.strip().split('\t')
    word = items[0].replace('(','').replace(')','')
    word_cats = items[1:]
    liwcD[word] = word_cats

def liwc(words):#words is a list of words
    global category_names, categoryIDs, liwcD, liwc_words
    total_words = len(words)
    line = ' '.join(words)
    feats = defaultdict(int)#keep track of liwc frequencies
    for word in sorted(liwcD.keys()): #first 9 words are emojis with special characters TODO: treat them separately
        cats = liwcD[word] #list of liwc categories
        if '*' in word:
            pattern = re.compile(' %s'%word.replace('*',''))
        else:
            pattern = re.compile(' %s '%word)
        matches = [(m.start(0), m.end(0)) for m in re.finditer(pattern, line)] #check is liwc word is in sentence
        if matches != []: #count matches
            for C in cats:
                feats[int(C)]+=len(matches)
        else:
            for C in cats:
                feats[int(C)] += 0
    if total_words != 0: #if 0 zero words in sentence - create zero vector
        liwc_features = [(float(feats[key])/total_words) for key in sorted(feats)]
    else:
        liwc_features = ','.join([0]*73)
    category_names = [categoryIDs[str(c)] for c in sorted(feats)]
    return liwc_features

if __name__=='__main__':
    ext=int(sys.argv[2])
    ext1=int(sys.argv[3])
    transcriptFiles = glob(sys.argv[1] + '[0-9][0-9][0-9]_P/[0-9][0-9][0-9]_TRANSCRIPT.csv')
    for i in range(ext,ext1):
        t = pd.read_csv(transcriptFiles[i], delimiter=',|\t')
        t = t.fillna("")
        participantNo=transcriptFiles[i][-18:-15]
        print participantNo
        for j in xrange(len(t)):
            if t.iloc[j]['speaker']=='Participant':
                utterance=re.search(".*\((.*)\)$", t.iloc[j]['value'])
                if utterance is not None:
                    utterance=utterance.group(1)
                else:
                    utterance=t.iloc[j]['value']
                utterance=utterance.strip()
                split_utterance=utterance.split(" ")
                feature=liwc(split_utterance)
                features.append([participantNo, utterance]+feature)

    with open('liwc_new'+str(ext)+'_'+str(ext1)+'.csv','w') as f:
        writer=csv.writer(f)
        writer.writerow(['video','question']+category_names)
        for item in features:
            writer.writerow(item)

