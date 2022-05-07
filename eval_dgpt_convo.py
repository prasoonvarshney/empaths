# edited from https://github.com/abaheti95/ToxiChat, train_and_evaluate_DGPT_offensive_classifier.py


# We will train offensive classifier on top of the DGPT model
# We will provide a dictionary of tasks to be trained upon in the arguments
# The model training and testing code will be implemented using the transformers library with pytorch backend

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import pdb

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer
from torch import nn
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
import json
import csv

# PRETRAINED_GPT2_MODEL = 'GPT2-base-cased'
PRETRAINED_GPT2_MODEL = 'microsoft/DialoGPT-medium'
# Other global constants required for the code
POSSIBLE_BATCH_SIZE = 1
MAX_SEQ_THRESH = 512

FILE = 'generations_toxichat_test_data_model'

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

class OC_S_offensive_Dataset(Dataset):
	def __init__(self):
		super(OC_S_offensive_Dataset, self).__init__()

		f = pd.read_csv(open(FILE + ".csv"))

		fi = []
		convid = []
		context = []
		idx = []
		pprompt = []
		sprompt = []
		ppl = []
		context_pre = []

		for row in f.itertuples():
			if not pd.isna(row[6]):
				idx.append(row[1])
				convid.append(row[2])
				context_pre.append(row[3])
				pprompt.append(row[4])
				sprompt.append(row[5])
				fi.append(row[6])
				ppl.append(row[7])
				context.append((row[3] + row[6]).split('<|endoftext|>'))

		self.instances = fi
		self.conv_ids = convid
		self.nsamples = len(self.instances)
		self.context = context
		self.idx = idx
		self.context_pre = context_pre
		self.pprompt = pprompt
		self.sprompt = sprompt
		self.ppl = ppl

	def __getitem__(self, index):
		return self.context[index]

	def __len__(self):
		return self.nsamples

def get_GPT2_string_from_utterances(utterances):
	# We will append EOS token after each utterance
	return ' EOS '.join([u.strip() for u in utterances]) + " EOS "

class OC_S_DGPT_TokenizeCollator():
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, batch):
		all_GPT2_model_input_texts = list()
		lengths = list()
		for i, data_dict in enumerate(batch):
			GPT2_string = get_GPT2_string_from_utterances(data_dict).replace(" EOS ", self.tokenizer.eos_token)
			all_GPT2_model_input_texts.append(GPT2_string)

			if lengths:
				lengths.append(len(data_dict) + lengths[-1])
			else:
				lengths.append(len(data_dict) - 1)

		# Tokenize
		all_GPT2_model_inputs_tokenized = self.tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
		input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
		try:
			assert input_ids.size(1) < 512
		except AssertionError:
			input_ids = input_ids[:, :512]
			input_ids[:, 511][input_ids[:, 511] != self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id

		# Extract the word_ids of CLS tokens i.e. the beginning of all the utterances
		eos_token_ids = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)

		# Convert the pad_token_ids to eos_token_ids as there is no pad token in DGPT model
		input_ids[input_ids==self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id
		
		
		return {"input_ids": input_ids, "lengths": lengths, "eos_token_ids": eos_token_ids, "input_str": all_GPT2_model_input_texts, "batch_data": batch}


class GPT2ForOC_S_offensive(GPT2LMHeadModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_off_labels = 2
		self.num_stance_labels = 3
		self.dropout = nn.Dropout(config.embd_pdrop)
		self.off_classifier = nn.Linear(config.hidden_size, self.num_off_labels)
		self.loss_fct = nn.CrossEntropyLoss()
	
	def forward(
		self,
		input_ids,
		utterance_eos_ids,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		off_labels=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
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
		outputs = self.transformer(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		# Type of outputs = BaseModelOutputWithPastAndCrossAttentions
		# ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
		GPT2_last_layer_output = outputs.last_hidden_state

		# Extract all EOS token representations from GPT2's last layer representations
		eos_token_representation = GPT2_last_layer_output[utterance_eos_ids[0], utterance_eos_ids[1], :]
		# Apply dropout on representations
		eos_token_representation = self.dropout(eos_token_representation)
		# Compute logits from cls representations
		off_logits = self.off_classifier(eos_token_representation)
		# target_logits = self.target_classifier(eos_token_representation)

		outputs = (off_logits,) + outputs[2:]
		# If off_labels given, compute loss from off_logits
		
		loss = 0.0
		if off_labels is not None:
			loss = self.loss_fct(off_logits.view(-1, self.num_off_labels), off_labels.view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)

def make_predictions_on_offensive_dataset(dataloader, model, tokenizer, device, dev_flag = False, threshold=0.5):
	# Create tqdm progressbar
	if not dev_flag:
		pbar = tqdm(dataloader)
	else:
		pbar = dataloader
	# Setting model to eval for predictions
	# NOTE: assuming that model is already in the given device
	model.eval()
	all_convs_str = list()
	all_off_prediction_scores = list()
	softmax_func = nn.Softmax(dim=1)
	with torch.no_grad():
		for step, batch in enumerate(pbar):
			all_convs_str.extend(batch["input_str"])
			# Create testing instance for model
			input_dict = {"input_ids": batch["input_ids"].to(device), "utterance_eos_ids": batch["eos_token_ids"]}
			logits = model(**input_dict)[0]

			off_logits = logits

			# Apply softmax on the off_logits			
			softmax_off_logits = softmax_func(off_logits)

			prediction_scores = softmax_off_logits[:, 1].cpu().tolist()

			lengths = batch["lengths"]

			all_off_prediction_scores.extend([prediction_scores[i] for i in lengths])
			
	return all_convs_str, all_off_prediction_scores

def main():

	df = pd.read_csv(open(FILE + '.csv'))

	#2.1 Initialize the collator with GPT2 tokenizer
	tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_GPT2_MODEL)
	tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	# tokenizer.pad_token = '[PAD]'
	# tokenizer.eos_token = '<|endoftext|>'
	# tokenizer.pad_token_id = 50257
	# tokenizer.eos_token_id = 50256
	tokenize_collator = OC_S_DGPT_TokenizeCollator(tokenizer)

	test_dataset = OC_S_offensive_Dataset()
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=tokenize_collator)

	model = GPT2ForOC_S_offensive.from_pretrained('/home/jessica/empaths/ToxiChat/saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e3')
	tokenizer = GPT2Tokenizer.from_pretrained('/home/jessica/empaths/ToxiChat/saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e3')
	model.to(device)

	# Put the model in evaluation mode. The dropout layers behave differently during evaluation.
	model.eval()

	# Test set evaluation
	threshold = json.load(open('ToxiChat/results/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e3/results.json'))["Off_task"]["best_dev_threshold"]

	test_str_convs, test_prediction_scores = make_predictions_on_offensive_dataset(test_dataloader, model, tokenizer, device, threshold=threshold)

	dgpt_csv = csv.writer(open(FILE + '_dgpt.csv', 'w'), delimiter='\t')
	dgpt_csv.writerow(['', 'conv_id', 'dialogue_context', 'perspective_prompt', 'strategy_prompt', 'generated_sentence', 'perplexity', 'toxic_score'])

	for i in range(len(test_prediction_scores)):
		dgpt_csv.writerow([test_dataset.idx[i], test_dataset.conv_ids[i], test_dataset.context_pre[i], test_dataset.pprompt[i], test_dataset.sprompt[i], test_dataset.instances[i], test_dataset.ppl[i], test_prediction_scores[i]])
		

if __name__ == '__main__':
	main()