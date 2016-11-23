import pandas as pd
import os,re,operator
import json
import sys

filenames = []
p=re.compile('.*_TRANSCRIPT.csv')
for (path,dir,files) in os.walk(sys.argv[1]):
    for each in files:
        if p.match(each):
            filenames.append(path+"/"+each)
print filenames

print len(filenames)
stopwords = ['okay','cool','nice','awesome','really','mhm','[laughter]']

data_file = open('dictionary.json','r')
questions = json.load(data_file)

for file in filenames:
    df = pd.read_csv(file,sep='\t')
    i = 0
    while i<len(df):
        if df.iloc[i]['value'].find("ask a few questions to get us started") > -1:
            break

        i += 1
    print i
    df = df[i+1:]
    if len(df) == 0:
        print "hiiiii"
        break
    i = 0
    while i < len(df):
        if df.iloc[i]['speaker'] == "Ellie":
            try:
                curr = i
                if df.iloc[i+1]['speaker'] == "Ellie":
                    curr = i+1
                    i += 1
                val = re.search(".*\((.*)\)$", df.iloc[curr]['value'])
                if val != None:
                    val = val.group(1)
                else:
                    val = df.iloc[curr]['value']
                if val not in stopwords:
                    if val in questions:
                        questions[val] = questions[val] + 1
                    else:
                        questions[val] = 1
            except:
                print "index out of bound"


        i += 1
data_file.close()

data_file = open('dictionary.json','w')
json.dump(questions,data_file)
data_file.close()

#print (questions)
# sorted_questions = sorted(questions.items(), key=operator.itemgetter(1),reverse=True)
# ques = open('questions.csv', 'wb')
# ques.write("Questions")
# ques.write(",")
# ques.write("Count")
# ques.write("\n")
# for q in sorted_questions:
#     ques.write(q[0])
#     ques.write(",")
#     ques.write(str(q[1]))
#     ques.write("\n")
