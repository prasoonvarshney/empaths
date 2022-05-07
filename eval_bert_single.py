# edited from https://github.com/abaheti95/ToxiChat, train_and_evaluate_BERT_offensive_classifier.py

# We will train single utterance offensive classifier on top of the BERT Large model
# We will provide a dictionary of tasks to be trained upon in the arguments (but probably we will only send OC_S offend data here)
# The model training and testing code will be implemented using the transformers library with pytorch backend

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import pdb

from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch

import random

import os
import re
import math
import time
import copy
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import metrics
import csv

# PRETRAINED_BERT_MODEL = 'Bert-base-cased'
PRETRAINED_BERT_MODEL = 'bert-large-cased'
# Other global constants required for the code
POSSIBLE_BATCH_SIZE = 1
MAX_SEQ_THRESH = 512

FILE = 'generations_toxichat_test_data_model'

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

class OC_S_Bert_offensive_TokenizeCollator():
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, batch):
		all_bert_model_input_texts = list()

		for i, data_dict in enumerate(batch):

			bert_string = f"[CLS] {data_dict} [SEP]"
			all_bert_model_input_texts.append(bert_string)

		# Tokenize
		all_Bert_model_inputs_tokenized = self.tokenizer.batch_encode_plus(all_bert_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
		input_ids, attention_mask = all_Bert_model_inputs_tokenized['input_ids'], all_Bert_model_inputs_tokenized['attention_mask']
		try:
			assert input_ids.size(1) < 512
		except AssertionError:
			input_ids = input_ids[:, :512]
			attention_mask = attention_mask[:, :512]

		# assert len(batch) == len(gold_off_labels), f"Assertion Failed, batch of size {len(batch)} != number of gold off labels {len(gold_off_labels)}"
		
		# Convert token_ids into tuples for future processing
		return {"input_str": all_bert_model_input_texts, "input_ids": input_ids, "attention_mask":attention_mask, "batch_data": batch}

class BertForOC_S_offensive(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_off_labels = 2
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.offensive_classifier = nn.Linear(config.hidden_size, self.num_off_labels)
		
		self.offensive_loss_fct = nn.CrossEntropyLoss()

	def forward(
		self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
		offensive_labels=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
			Classification (or regression if config.num_labels==1) loss.
		logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
		"""
		outputs = self.bert(
			input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
		)
		# Type of outputs = BaseModelOutputWithPastAndCrossAttentions
		# ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
		# ref2: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html in class BertForSequenceClassification(BertPreTrainedModel):
		cls_token_representation = outputs[1]
		# Apply dropout
		offensive_classifier_input = self.dropout(cls_token_representation)
		# Compute stance logits from concatenated eos representations
		offensive_logits = self.offensive_classifier(offensive_classifier_input)

		outputs = (offensive_logits,) + outputs[2:]
		# If offensive_labels given, compute loss from offensive_logits
		
		loss = 0.0
		if offensive_labels is not None:
			loss = self.offensive_loss_fct(offensive_logits.view(-1, self.num_off_labels), offensive_labels.view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)

def make_bert_predictions_on_offensive_dataset(dataloader, model, tokenizer, device, segment_name, dev_flag = False, threshold=0.5):
	# Create tqdm progressbar
	if not dev_flag:
		pbar = tqdm(dataloader)
	else:
		pbar = dataloader
	# Setting model to eval for predictions
	# NOTE: assuming that model is already in the given device
	model.eval()
	all_convs_str = list()
	all_off_predictions = list()
	all_off_prediction_scores = list()
	softmax_func = nn.Softmax(dim=1)
	with torch.no_grad():
		for step, batch in enumerate(pbar):
			all_convs_str.extend(batch["input_str"])
			# Create testing instance for model
			input_dict = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
			logits = model(**input_dict)[0]

			off_logits = logits

			# Apply softmax on the off_logits			
			softmax_off_logits = softmax_func(off_logits)

			prediction_scores = softmax_off_logits.cpu().tolist()

			all_off_prediction_scores.extend(prediction_scores)
			
	return all_convs_str, all_off_prediction_scores

class OC_S_offensive_Dataset(Dataset):
	def __init__(self):
		super(OC_S_offensive_Dataset, self).__init__()

		f = pd.read_csv(open(FILE + ".csv"))

		idx = []
		convid = []
		context = []
		pprompt = []
		sprompt = []
		gen = []
		ppl = []

		for row in f.itertuples():
			if not pd.isna(row[6]):
				idx.append(row[1])
				convid.append(row[2])
				context.append(row[3])
				pprompt.append(row[4])
				sprompt.append(row[5])
				gen.append(row[6])
				ppl.append(row[7])

		self.instances = gen
		self.conv_ids = convid
		self.nsamples = len(self.instances)
		self.idx = idx
		self.context = context
		self.pprompt = pprompt
		self.sprompt = sprompt
		self.ppl = ppl

	def __getitem__(self, index):
		return self.instances[index]

	def __len__(self):
		return self.nsamples

def main():
	test_dataset = OC_S_offensive_Dataset()

	# #2.1 Initialize the collator with Bert tokenizer
	tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
	tokenize_collator = OC_S_Bert_offensive_TokenizeCollator(tokenizer)

	test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=tokenize_collator)

    # # Load from a previously trained model
	model = BertForOC_S_offensive.from_pretrained('/home/jessica/empaths/ToxiChat/saved_models/OC_S_post_thread/BERT_large_OC_S_offensive_e8')
	tokenizer = BertTokenizer.from_pretrained('/home/jessica/empaths/ToxiChat/saved_models/OC_S_post_thread/BERT_large_OC_S_offensive_e8')
	model.to(device)

	# # Put the model in evaluation mode. The dropout layers behave differently during evaluation.
	model.eval()

    # # Evaluate on test

	# 0 = not toxic, 1 = toxic (offensive)
	# to get toxic score, we take index 1 of the test_prediction_scores

	test_str_convs, test_prediction_scores = make_bert_predictions_on_offensive_dataset(test_dataloader, model, tokenizer, device, "dev", True)
	
	bert_csv = csv.writer(open(FILE + '_bert.csv', 'w'), delimiter='\t')
	bert_csv.writerow(['', 'conv_id', 'dialogue_context', 'perspective_prompt', 'strategy_prompt', 'generated_sentence', 'perplexity', 'toxic_score'])

	for i in range(len(test_prediction_scores)):
		bert_csv.writerow([test_dataset.idx[i], test_dataset.conv_ids[i], test_dataset.context[i], test_dataset.pprompt[i], test_dataset.sprompt[i], test_dataset.instances[i], test_dataset.ppl[i], test_prediction_scores[i][1]])
		

if __name__ == '__main__':
	main()