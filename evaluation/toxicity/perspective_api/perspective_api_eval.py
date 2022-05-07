import numpy as np
import pandas as pd
import sys
import time

from googleapiclient import discovery
from tqdm import tqdm
from os.path import exists

class PerspectiveAPIScoreEval:

    def __init__(self):

        self.API_KEYS = ['AIzaSyAOf7k6AwhH3MRxCiWJNeHUEe49_Lck-J0',
                         'AIzaSyBVjqmVjaYE916cbMQhF1CJErfCVdqRHpU',
                         'AIzaSyAxGk7xFc_WAGytqxWQBdKR6LYxveCpRaU']

        self.persp_clients = []
        for key in self.API_KEYS:
            persp_client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

            self.persp_clients.append(persp_client)

        self.counter = 0
        self.file_save_freq = 100

    def load_tsv(self, filename):
        df = pd.read_csv(filename, sep='\t')
        return df

    def evaluate_toxicity_perspective_api(self, input_text, conv_id, utterance_id):
        analyze_request_emp = {
            'comment': {'text': input_text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        client = self.persp_clients[self.counter % len(self.persp_clients)]
        persp_score = np.nan

        try:
            response = client.comments().analyze(
                body=analyze_request_emp).execute()
            persp_score = float(
                response['attributeScores']['TOXICITY']['summaryScore'][
                    'value'])
        except Exception as e:
            print("Error while calculating Perspective API score for conv : "
                  + str(conv_id) + " utterance id : " + str(utterance_id))

        self.counter += 1

        return persp_score

    def save_intermediate_result_to_file(self, dst_df, dst_file):
        dst_df.to_csv(dst_file, sep="\t", index=False)

    def add_perspective_api_score(self, dst_df, dst_file, score_col_pairs):
        for index, row in tqdm(dst_df.iterrows(), total=dst_df.shape[0]):
            if row['speaker_idx'] == 1:
                for col, score_col in score_col_pairs.items():
                    if np.isnan(row[score_col]) and isinstance(row[col], str):
                        persp_score = self.evaluate_toxicity_perspective_api(
                            row[col], row['conv_id'], row['utterance_idx'])
                        # print(persp_score)
                        # dst_df.set_value(index, score_col, persp_score)
                        dst_df.at[index, score_col] = persp_score
                        # adding delay due to perspective API constraints
                        time.sleep(0.5)

            if self.counter % self.file_save_freq == 0 and self.counter != 0:
                print("Saving to file ...")
                self.save_intermediate_result_to_file(dst_df, dst_file)

        self.save_intermediate_result_to_file(dst_df, dst_file)

    def get_dst_df(self, src_df, dst_file):
        dst_df = None
        if exists(dst_file):
            dst_df = self.load_tsv(dst_file)
        else:
            dst_df = src_df.copy()
        # print(dst_df.columns)
        # dst_df.set_index("Unnamed: 0", inplace=True)

        return dst_df

    def persp_scores_helper(self, src_file, dst_file, columns_to_evaluate):
        src_df = self.load_tsv(src_file)

        # get dst df
        dst_df = self.get_dst_df(src_df, dst_file)

        # add the new columns required for storing perspective api scores
        score_col_pairs = dict()
        for col_name in columns_to_evaluate:
            score_col_name = 'persp_api_sc_' + col_name
            score_col_pairs[col_name] = score_col_name
            if score_col_name not in dst_df.columns:
                dst_df[score_col_name] = np.nan
            else:
                print("Column with the " + score_col_name +
                      " name already exists! Not creating a new one.")

        # evaluate and add persp api scores
        self.add_perspective_api_score(dst_df, dst_file, score_col_pairs)


if __name__ == "__main__":
    src_file = sys.argv[1]
    dst_file = sys.argv[2]

    columns_to_evaluate = ['generated_sentence']

    persp_eval = PerspectiveAPIScoreEval()

    persp_eval.persp_scores_helper(src_file, dst_file, columns_to_evaluate)
