import os
import json
import sys
import torch
import random
import numpy as np
import spacy

import argparse

parser = argparse.ArgumentParser(description='File path of preprocessed data')
parser.add_argument('--drive_path', type=str, default='',help='Path for running the code.')
parser.add_argument('--save_pkl_path', type=str, default='', help='Path for saving preprocessed data for TRIP with dependency graphs and ConceptNet Numberbatch features')
parser.add_argument('--cn_nb_path', type=str, default='',help='Path for ConceptNet Numberbatch embedding file.')
parser.add_argument('--jar_path', type=str, default='',help='stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar')
parser.add_argument('--models_jar_path', type=str, default='',help='stanford-corenlp-4.2.2-models-english.jar')


args = parser.parse_args()

DRIVE_PATH = args.drive_path
# mode = 'bert' # BERT large
mode = 'roberta' # RoBERTa large
# mode = 'roberta_mnli' # RoBERTa large pre-trained on MNLI
# mode = 'deberta' # DeBERTa base for training on TRIP
# mode = 'deberta_large' # DeBERTa large for training on CE and ART

task_name = 'trip'
# task_name = 'ce'
# task_name = 'art'

debug = False
# debug = True


config_batch_size = 1
config_lr = 1e-5 # Selected learning rate for best RoBERTa-based model in TRIP paper
config_epochs = 15
# config_epochs = 1


# Loss weights for (attributes, preconditions, effects, conflicts, story choices)
if task_name != 'trip':
  print("We do not need a loss weighting scheme for %s dataset. Ignoring this cell." % task_name)

sys.path.append(DRIVE_PATH)

if task_name in ['trip', 'ce']:
  multiple_choice = False
elif task_name == 'art':
  multiple_choice = True
else:
  raise ValueError("Task name should be set to 'trip', 'ce', or 'art' in the first cell of the notebook!")

if mode == 'bert':
  model_name = 'bert-large-uncased'
elif mode == 'roberta':
  model_name = 'roberta-large'
elif mode == 'roberta_mnli':
  model_name = 'roberta-large-mnli'
elif mode == 'deberta':
  model_name = 'microsoft/deberta-base'
elif mode == 'deberta_large':
  model_name = 'microsoft/deberta-large'



from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, AlbertTokenizer, T5Tokenizer, GPT2Tokenizer

from DeBERTa import deberta
if mode in ['bert']:
  tokenizer_class = BertTokenizer
elif mode in ['roberta', 'roberta_mnli']:
  tokenizer_class = RobertaTokenizer
elif mode in ['deberta', 'deberta_large']:
  tokenizer_class = DebertaTokenizer

tokenizer = tokenizer_class.from_pretrained(model_name, 
                                                do_lower_case = False, 
                                                cache_dir=os.path.join(DRIVE_PATH, 'cache'))

from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DebertaForSequenceClassification, AlbertForSequenceClassification, AdamW
from transformers import BertForMultipleChoice, RobertaForMultipleChoice, AlbertForMultipleChoice, DebertaModel
from transformers import BertModel, RobertaModel, AlbertModel, DebertaModel, T5Model, T5EncoderModel, GPT2Model
from transformers import RobertaForMaskedLM
from transformers import BertConfig, RobertaConfig, DebertaConfig, AlbertConfig, T5Config, GPT2Config
from www.model.transformers_ext import DebertaForMultipleChoice
from torch.optim import Adam
if not multiple_choice:
  if mode == 'bert':
    model_class = BertForSequenceClassification
    config_class = BertConfig
    emb_class = BertModel
  elif mode in ['roberta', 'roberta_mnli']:
    model_class = RobertaForSequenceClassification
    config_class = RobertaConfig
    emb_class = RobertaModel
    lm_class = RobertaForMaskedLM
  elif mode in ['deberta', 'deberta_large']:
    model_class = DebertaForSequenceClassification
    config_class = DebertaConfig
    emb_class = DebertaModel
else:
  if mode == 'bert':
    model_class = BertForMultipleChoice
    config_class = BertConfig
    emb_class = BertModel    
  elif mode in ['roberta', 'roberta_mnli']:
    model_class = RobertaForMultipleChoice
    config_class = RobertaConfig
    emb_class = RobertaModel
    lm_class = RobertaForMaskedLM
  elif mode in ['deberta', 'deberta_large']:
    model_class = DebertaForMultipleChoice
    config_class = DebertaConfig
    emb_class = DebertaModel


from www.utils import print_dict

partitions = ['train', 'dev', 'test']
subtasks = ['cloze', 'order']

# We can split the data into multiple json files later
data_file = os.path.join(DRIVE_PATH, 'all_data/www.json')
with open(data_file, 'r') as f:
  dataset = json.load(f)

print('Preprocessed examples:')
for ex_idx in [0,1,5,10]:
  ex = dataset['dev'][list(dataset['dev'].keys())[ex_idx]]
  print_dict(ex)


cloze_dataset = {p: [] for p in dataset}
order_dataset = {p: [] for p in dataset}

for p in dataset:
  for exid in dataset[p]:
    ex = dataset[p][exid]

    if ex['type'] == None:
      continue
    
    ex_plaus = dataset[p][str(ex['story_id'])]

    if ex['type'] == 'cloze':
      cloze_dataset[p].append(ex)
      cloze_dataset[p].append(ex_plaus) # For every implausible story, add a copy of its corresponding plausible story

    # Exclude augmented ordering examples from dev and test, since the breakpoints aren't always accurate in those
    elif ex['type'] == 'order' and not (p != 'train' and ex['aug']): 
      order_dataset[p].append(ex)
      order_dataset[p].append(ex_plaus)


from www.utils import print_dict
import json
from collections import Counter

data_file = os.path.join(DRIVE_PATH, 'all_data/www_2s_new.json')
with open(data_file, 'r') as f:
  cloze_dataset_2s, order_dataset_2s = json.load(f)  

for p in cloze_dataset_2s:
  label_dist = Counter([ex['label'] for ex in cloze_dataset_2s[p]])
  print('Cloze label distribution (%s):' % p)
  print(label_dist.most_common())
print_dict(cloze_dataset_2s['train'][0])

if task_name != 'trip':
  raise ValueError('Please configure task_name in first cell to "trip" to run TRIP results!')

from www.dataset.prepro import get_tiered_data, balance_labels
from www.dataset.featurize import add_bert_features_tiered, get_tensor_dataset_tiered
from collections import Counter

tiered_dataset = cloze_dataset_2s

# Debug the code on a small amount of data
if debug:
  for k in tiered_dataset:
    tiered_dataset[k] = tiered_dataset[k][:20]

if debug:
  for k in tiered_dataset:
    tiered_dataset[k] = tiered_dataset[k][:12]

# train_spans = True
train_spans = False
if train_spans:
  tiered_dataset = get_story_spans_2s(tiered_dataset, train_only=True)
  tiered_dataset['train'] = [ex for ex in tiered_dataset['train'] if ex['label'] != -1] # For now, ignore examples where both stories are plausible :(

seq_length = 16 # Max sequence length to pad to

tiered_dataset = get_tiered_data(tiered_dataset)
tiered_dataset = add_bert_features_tiered(tiered_dataset, tokenizer, seq_length, add_segment_ids=True)

import gensim
import torch
import numpy as np
from www.dataset.featurize import add_dependency_graph

cn_nb = gensim.models.KeyedVectors.load_word2vec_format(args.cn_nb_path, binary=False)
tiered_dataset = add_dependency_graph(tiered_dataset,cn_nb,jar_path=args.jar_path, models_jar_path=args.models_jar_path)

for p in tiered_dataset:
    for ex in tiered_dataset[p]:
        for story in ex['stories']:
            for i in range(len(story['label_list'])):
                story['label_list'][i] = {v: k for k, v in story['label_list'][i].items()}

import pickle
file_to_write = open(args.save_pkl_path, "wb")
pickle.dump(tiered_dataset, file_to_write)






