import csv

name = 'generations_toxichat_test_data_model_persp'

f = csv.reader(open(name + '.tsv'), delimiter='\t')

fi = csv.writer(open(name + '.csv', 'w'))

for row in f:
    fi.writerow(row)