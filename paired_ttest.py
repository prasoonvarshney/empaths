from scipy import stats
import pandas as pd

FILE = "generations_toxichat_test_data_model_huggingface.csv"
# FILE = "generations_toxichat_test_data_model_persp.tsv"
threshold = 0.7
# toxic = 'toxic_score'
toxic = 'toxicity'
# toxic='persp_api_sc_generated_sentence'

f = pd.read_csv(open(FILE), delimiter='\t')

df2 = f[f[toxic] >= threshold]

strategies = ['<|acknowledging|>', '<|agreeing|>', '<|consoling|>', '<|encouraging|>', '<|questioning|>', '<|suggesting|>', '<|sympathizing|>', '<|wishing|>', 'Autopicked by Finetuned Model']
# filter by strategy + pretrained model
for strategy in strategies:
    df3 = df2[df2['strategy_prompt'] == strategy][toxic].values
    df4 = df2[df2['strategy_prompt'] == 'None - Pretrained Model'][toxic].values

    try:
        print(strategy)
        print(df3)
        print('strategy', sum(df3)/len(df3))
        print('pretrained', sum(df4)/len(df4))
        print(stats.ttest_ind(df3, df4))
    except ZeroDivisionError:
        pass