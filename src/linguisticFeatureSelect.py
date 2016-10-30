import csv
import scipy.stats

depressed=[]
not_depressed=[]

with open('data/liwc_depressed.csv') as f:
	reader=csv.reader(f)
	for row in reader:
		depressed.append(row[1:])

with open('data/liwc_notdepressed.csv') as f:
	reader=csv.reader(f)
	for row in reader:
		not_depressed.append(row[1:])

d_f=[[row[i] for row in depressed] for i in range(len(depressed[0]))]
nd_f=[[row[i] for row in not_depressed] for i in range(len(not_depressed[0]))]

for i in range(0,len(d_f)):
	for j in range(0,len(d_f[0])):
		d_f[i][j]=float(d_f[i][j])

for i in range(0,len(nd_f)):
	for j in range(0,len(nd_f[0])):
		nd_f[i][j]=float(nd_f[i][j])

features=[]
for i in range(1,len(d_f)):
	t,p=scipy.stats.ttest_ind(d_f[i], nd_f[i], None, False)
	if p<=0.10:
		features.append(i)


print len(features), features
