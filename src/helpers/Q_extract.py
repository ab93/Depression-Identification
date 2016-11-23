import pandas as pd
import os,re
import numpy as np
filenames = []
p=re.compile('.*_TRANSCRIPT.csv')
for (path,dir,files) in os.walk('../../data/'):
    for each in files:
        if p.match(each):
            filenames.append(path+"/"+each)
print filenames

print len(filenames)

list_of_qs = ["what are you","what do you do now"]
#print len(list_of_qs)
#for each in list_of_qs:
#    print each

#data_file = open('dictionary.json','r')
#questions = json.load(data_file)
#print questions

for each in list_of_qs:
    print "Q: ",each
    for file in filenames:
        df = pd.read_csv(file,sep='\t')
        df = df.replace(np.nan,"",regex=True)
        i = 0
        found = 0
        while i < len(df):
            val = re.search(r".*\((.*)\)$",df.iloc[i]['value'])
            if val != None:
                val = val.group(1)
            else:
                val = df.iloc[i]['value']
            if df.iloc[i]['speaker'] == "Ellie" and val==each:
                context = ""
                beg=i-5 if i-5>=0 else 0
                end=i+5 if i+5<len(df) else len(df)-1
                print "File: ",file
                print df.iloc[beg:end][['speaker','value']]
                inp=raw_input()
                if inp !='y':
                    found = 1
                    break
            i += 1
        if found==1:
            break