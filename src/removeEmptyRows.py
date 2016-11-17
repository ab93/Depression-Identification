import csv

input = open('../data/discriminativeFACET_o.csv', 'rb')
output = open('../data/discriminativeFACET.csv', 'wb')

writer = csv.writer(output)

for row in csv.reader(input):

    if row or any(row) or any(field.strip() for field in row):
        #print row
        writer.writerow(row)

input.close()

output.close()