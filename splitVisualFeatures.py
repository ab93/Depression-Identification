import csv

depressed=[]
not_depressed=[]
count=0
instance=1
with open('training_split.csv','r') as f:
	reader=csv.reader(f)
	reader.next()
	for row in reader:
		count+=1
		if row[1]=='0':
			not_depressed.append(row[0])
		else:
			depressed.append(row[0])


d_f=open('liwc_d.csv','w')
nd_f=open('liwc_nd.csv','w')
d_csv=csv.writer(d_f)
nd_csv=csv.writer(nd_f)
with open('liwc.csv','r') as f:
	reader=csv.reader(f)
	reader.next()
	for row in reader:
		if instance<12179:
			if row[0] in depressed:
				d_csv.writerow([row[0]]+row[2:7]+row[8:])
			elif row[0] in not_depressed:
				nd_csv.writerow([row[0]]+row[2:7]+row[8:])

d_f.close()
nd_f.close()

