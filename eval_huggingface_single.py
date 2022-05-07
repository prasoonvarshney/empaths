# https://github.com/unitaryai/detoxify

from detoxify import Detoxify
import pandas
import csv

name = 'generations_toxichat_test_data_model'
f = pandas.read_csv(open(name + '.csv'))
scores = csv.writer(open(name + '_huggingface.csv', 'w'), delimiter='\t')

ex = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']

scores.writerow(['', 'conv_id', 'dialogue_context', 'perspective_prompt', 'strategy_prompt', 'generated_sentence', 'perplexity', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'])

idx = []
convid = []
context = []
pprompt = []
sprompt = []
gen = []
ppl = []

for row in f.itertuples():
    if not pandas.isna(row[6]):
        idx.append(row[1])
        convid.append(row[2])
        context.append(row[3])
        pprompt.append(row[4])
        sprompt.append(row[5])
        gen.append(row[6])
        ppl.append(row[7])

for i in range(0, len(idx), 100):
    print(i)
    toxic = Detoxify('original').predict(gen[i:i+100])

    for j in range(len(gen[i:i+100])):
        final_write = [idx[i+j], convid[i+j], context[i+j], pprompt[i+j], sprompt[i+j], gen[i+j], ppl[i+j]]

        for x in ex:
            final_write.append(toxic[x][j])
        
        scores.writerow(final_write)