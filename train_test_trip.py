import os
import json
import sys
import torch
import random
import numpy as np
import spacy
from tqdm import tqdm
from www.utils import print_dict
import json
from collections import Counter
from www.dataset.prepro import get_tiered_data, balance_labels
from www.dataset.featurize import add_bert_features_tiered, get_tensor_dataset_tiered
from collections import Counter

DRIVE_PATH = '/scratch/drjieliu_root/drjieliu/zheyuz/Verifiable-Coherent-NLU/'
sys.path.append(DRIVE_PATH)

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
config_epochs = 10
# config_epochs = 2


# Loss weights for (attributes, preconditions, effects, conflicts, story choices)
if task_name != 'trip':
  print("We do not need a loss weighting scheme for %s dataset. Ignoring this cell." % task_name)
# loss_weights = [0.0, 0.4, 0.4, 0.1, 0.1] # "All losses"
loss_weights = [0.0, 0.4, 0.4, 0.2, 0.0] # "Omit story choice loss"



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
import pickle
# pickle_file_debug = open('/scratch/drjieliu_root/drjieliu/zheyuz/Verifiable-Coherent-NLU/tiered_dataset_debug_new.pickle','rb')
pickle_file_debug = open('/scratch/drjieliu_root/drjieliu/zheyuz/Verifiable-Coherent-NLU/tiered_dataset_all_new.pickle','rb')
while True:
    try:
        tiered_dataset = pickle.load(pickle_file_debug)
    except EOFError:
        break
pickle_file_debug.close()

# train_spans = True
train_spans = False
# if train_spans:
#   tiered_dataset = get_story_spans_2s(tiered_dataset, train_only=True)
#   tiered_dataset['train'] = [ex for ex in tiered_dataset['train'] if ex['label'] != -1] # For now, ignore examples where both stories are plausible :(

seq_length = 16 # Max sequence length to pad to

# tiered_dataset = get_tiered_data(tiered_dataset)
# tiered_dataset = add_bert_features_tiered(tiered_dataset, tokenizer, seq_length, add_segment_ids=True)

import gensim
from nltk.parse.stanford import StanfordDependencyParser
# from www.dataset.featurize_my import add_dependency_graph

cn_nb = gensim.models.KeyedVectors.load_word2vec_format('/scratch/drjieliu_root/drjieliu/zheyuz/Verifiable-Coherent-NLU/numberbatch-en-19.08.txt', binary=False)
# tiered_dataset = add_dependency_graph(tiered_dataset,cn_nb)

tiered_tensor_dataset = {}
label_entities = {}
dep_graphs = {}
graph_labels = {}
max_story_length = max([len(ex['stories'][0]['sentences']) for p in tiered_dataset for ex in tiered_dataset[p]])
for p in tiered_dataset:
  tiered_tensor_dataset[p], label_entities[p], dep_graphs[p], graph_labels[p] = get_tensor_dataset_tiered(tiered_dataset[p], max_story_length, add_segment_ids=True)

from www.dataset.ann import att_to_idx, att_to_num_classes, att_types

subtask = 'cloze'
batch_sizes = [config_batch_size]
learning_rates = [config_lr]
epochs = config_epochs
eval_batch_size = 1
generate_learning_curve = False # Generate data for training curve figure in TRIP paper
# generate_learning_curve = True # Generate data for training curve figure in TRIP paper

num_state_labels = {}
for att in att_to_idx:
  if att_types[att] == 'default':
    num_state_labels[att_to_idx[att]] = 3
  else:
    num_state_labels[att_to_idx[att]] = att_to_num_classes[att] # Location attributes fall into this since they don't have well-define pre- and post-condition yet

# Ablation options:
# - attributes: skip attribute prediction phase
# - embeddings: DON'T input contextual embeddings to conflict detector
# - states: DON'T input states to conflict detector
# - states-labels: in states input to conflict detector, include predicted labels
# - states-logits: in states input to conflict detector, include state logits (preferred)
# - states-teacher-forcing: train conflict detector on ground truth state labels (not predictions)
# - states-attention: re-weight input to conflict detector with weights conditioned on states representation
ablation = ['attributes', 'states-logits'] # This is the default mode presented in the paper



from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from www.model.train import train_epoch_tiered
from www.model.eval import evaluate_tiered, save_results, save_preds, add_entity_attribute_labels
from sklearn.metrics import accuracy_score, f1_score
from www.utils import print_dict, get_model_dir
from www.model.transformers_ext import TieredModelPipeline
from www.dataset.ann import att_to_num_classes
import shutil
import pandas as pd

seed_val = 22 # Save random seed for reproducibility
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll keep the validation data here with a constant eval batch size
dev_sampler = SequentialSampler(tiered_tensor_dataset['dev'])
dev_dataloader = DataLoader(tiered_tensor_dataset['dev'], sampler=dev_sampler, batch_size=eval_batch_size)
dev_dataset_name = subtask + '_%s_dev'
dev_ids = [ex['example_id'] for ex in tiered_dataset['dev']]

all_losses = []
param_combos = []
combo_names = []
all_val_objs = []
output_dirs = []
best_obj = 0.0
best_model = '<none>'
best_dir = ''
best_obj2 = 0.0
best_model2 = '<none>'
best_dir2 = ''
feature_size = cn_nb['heat'].shape[0]
np.random.seed(0)
print('Beginning grid search for the %s sub-task over %s parameter combination(s)!' % (subtask, str(len(batch_sizes) * len(learning_rates))))
for bs in batch_sizes:
  for lr in learning_rates:
    print('\nTRAINING MODEL: bs=%s, lr=%s' % (str(bs), str(lr)))

    loss_values = []
    obj_values = []

    # Set up training dataset with new batch size
    train_sampler = SequentialSampler(tiered_tensor_dataset['train'])
    train_dataloader = DataLoader(tiered_tensor_dataset['train'], sampler=train_sampler, batch_size=bs)
    # random_ind = np.random.choice(len(tiered_tensor_dataset['train']), len(tiered_tensor_dataset['train']),replace=False)
    # tiered_tensor_dataset_train = tiered_tensor_dataset['train'][random_ind]
    # dep_graphs_train = dep_graphs['train']
    # label_entities_train = label_entities['train']
    # graph_labels_train = graph_labels['train']

    # Set up model
    config = config_class.from_pretrained(model_name,
                                          cache_dir=os.path.join(DRIVE_PATH, 'cache'))    
    emb = emb_class.from_pretrained(model_name,
                                          config=config,
                                          cache_dir=os.path.join(DRIVE_PATH, 'cache'))    
    if torch.cuda.is_available():
      emb.cuda()
    device = emb.device
    max_story_length = max([len(ex['stories'][0]['sentences']) for p in tiered_dataset for ex in tiered_dataset[p]])
    
    model = TieredModelPipeline(emb, max_story_length, len(att_to_num_classes), num_state_labels,
                                config_class, model_name, device, feature_size,
                                ablation=ablation, loss_weights=loss_weights).to(device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = total_steps)

    train_lc_data = []
    val_lc_data = []
    for epoch in tqdm(range(epochs)):
      # Train the model for one epoch
      print('[%s] Beginning epoch...' % str(epoch))

      epoch_loss, _ = train_epoch_tiered(model, optimizer, train_dataloader, device, seg_mode=False, 
                                         build_learning_curves=generate_learning_curve, val_dataloader=dev_dataloader, 
                                         train_lc_data=train_lc_data, val_lc_data=val_lc_data,dep_graphs=dep_graphs,label_entities=label_entities,graph_labels=graph_labels,cn_nb=cn_nb)
      
      # Save loss
      loss_values.append(epoch_loss)

      # Validate on dev set
      validation_results = evaluate_tiered(model, dev_dataloader, device, [(accuracy_score, 'accuracy'), (f1_score, 'f1')], seg_mode=False, return_explanations=True,dep_graphs=dep_graphs['dev'],label_entities=label_entities['dev'],graph_labels=graph_labels['dev'],cn_nb=cn_nb)
      metr_attr, all_pred_atts, all_atts, \
      metr_prec, all_pred_prec, all_prec, \
      metr_eff, all_pred_eff, all_eff, \
      metr_conflicts, all_pred_conflicts, all_conflicts, \
      metr_stories, all_pred_stories, all_stories, explanations = validation_results[:16]
      explanations = add_entity_attribute_labels(explanations, tiered_dataset['dev'], list(att_to_num_classes.keys()))

      print('[%s] Validation results:' % str(epoch))
      print('[%s] Preconditions:' % str(epoch))
      print_dict(metr_prec)
      print('[%s] Effects:' % str(epoch))
      print_dict(metr_eff)
      print('[%s] Conflicts:' % str(epoch))
      print_dict(metr_conflicts)
      print('[%s] Stories:' % str(epoch))
      print_dict(metr_stories)

      # Save accuracy - want to maximize verifiability of tiered predictions
      ver = metr_stories['verifiability']
      acc = metr_stories['accuracy']
      obj_values.append(ver)
      
      # Save model checkpoint
      print('[%s] Saving model checkpoint...' % str(epoch))
      model_param_str = get_model_dir(model_name.replace('/', '-'), subtask, bs, lr, epoch) + '_' +  '-'.join([str(lw) for lw in loss_weights]) +  '_tiered_pipeline_lc'
      if train_spans:
        model_param_str += 'spans'
      if len(model.ablation) > 0:
        model_param_str += '_ablate_'
        model_param_str += '_'.join(model.ablation)
      output_dir = os.path.join(DRIVE_PATH, 'saved_models_complex', model_param_str)
      output_dirs.append(output_dir)
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      save_results(metr_attr, output_dir, dev_dataset_name % 'attributes')
      save_results(metr_prec, output_dir, dev_dataset_name % 'preconditions')
      save_results(metr_eff, output_dir, dev_dataset_name % 'effects')
      save_results(metr_conflicts, output_dir, dev_dataset_name % 'conflicts')
      save_results(metr_stories, output_dir, dev_dataset_name % 'stories')
      save_results(explanations, output_dir, dev_dataset_name % 'explanations')

      # Just save story preds
      save_preds(dev_ids, all_stories, all_pred_stories, output_dir, dev_dataset_name % 'stories')

      emb = emb.module if hasattr(emb, 'module') else emb
      emb.save_pretrained(output_dir)
      torch.save(model, os.path.join(output_dir, 'classifiers.pth'))
      tokenizer.save_vocabulary(output_dir)

      if ver > best_obj:
        best_obj = ver
        best_model = model_param_str
        best_dir = output_dir
      if acc > best_obj2:
        best_obj2 = acc
        best_model2 = model_param_str
        best_dir2 = output_dir        

      for od in output_dirs:
        if od != best_dir and od != best_dir2 and os.path.exists(od):
          shutil.rmtree(od)

      print('[%s] Finished epoch.' % str(epoch))

    all_losses.append(loss_values)
    all_val_objs.append(obj_values)
    param_combos.append((bs, lr))
    combo_names.append('bs=%s, lr=%s' % (str(bs), str(lr)))

print('Finished grid search! :)')
print('Best validation *verifiability* %s from model %s.' % (str(best_obj), best_model))
print('Best validation *accuracy* %s from model %s.' % (str(best_obj2), best_model2))

if generate_learning_curve:
  print('Saving learning curve data...')
  train_lc_data = [subrecord for record in train_lc_data for subrecord in record] # flatten
  val_lc_data = [subrecord for record in val_lc_data for subrecord in record] # flatten

  train_lc_data = pd.DataFrame(train_lc_data)
  print(os.path.join(best_dir if best_dir != '<none>' else best_dir2, 'learning_curve_data_train.csv'))
  train_lc_data.to_csv(os.path.join(best_dir if best_dir != '' else best_dir2, 'learning_curve_data_train.csv'), index=False)
  val_lc_data = pd.DataFrame(val_lc_data)
  val_lc_data.to_csv(os.path.join(best_dir if best_dir != '' else best_dir2, 'learning_curve_data_val.csv'), index=False)
  print('Learning curve data saved. %s rows saved for training, %s rows saved for validation.' % (str(len(train_lc_data.index)), str(len(val_lc_data.index))))

# import shutil

# # Delete non-best model checkpoints
# for od in output_dirs:
#   if od != best_dir and od != best_dir2 and os.path.exists(od):
#     shutil.rmtree(od)

for layer in model.precondition_classifiers:
  layer.eval()
for layer in model.effect_classifiers:
  layer.eval()


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from www.model.eval import evaluate_tiered, save_results, save_preds, list_comparison, add_entity_attribute_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
metrics = [(accuracy_score, 'accuracy'), (precision_score, 'precision'), (recall_score, 'recall'), (f1_score, 'f1')]
import numpy as np
from www.utils import print_dict

# print('Testing model: %s.' % probe_model)

print('Testing model ')

# May alter this depending on which partition(s) you want to run inference on
for p in tiered_dataset:
  if p != 'test':
    continue

  p_dataset = tiered_dataset[p]
  p_tensor_dataset = tiered_tensor_dataset[p]
  p_sampler = SequentialSampler(p_tensor_dataset)
  p_dataloader = DataLoader(p_tensor_dataset, sampler=p_sampler, batch_size=1)
  dev_dataset_name = subtask + '_%s_' + p
  p_ids = [ex['example_id'] for ex in tiered_dataset[p]]

  # Get preds and metrics on this partition
  metr_attr, all_pred_atts, all_atts, \
  metr_prec, all_pred_prec, all_prec, \
  metr_eff, all_pred_eff, all_eff, \
  metr_conflicts, all_pred_conflicts, all_conflicts, \
  metr_stories, all_pred_stories, all_stories, explanations = evaluate_tiered(model, p_dataloader, device, [(accuracy_score, 'accuracy'), (f1_score, 'f1')], seg_mode=False, return_explanations=True,dep_graphs=dep_graphs[p],label_entities=label_entities[p],graph_labels=graph_labels[p],cn_nb=cn_nb)
  explanations = add_entity_attribute_labels(explanations, tiered_dataset[p], list(att_to_num_classes.keys()))

  # save_results(metr_attr, output_dir, dev_dataset_name % 'attributes')
  # save_results(metr_prec, output_dir, dev_dataset_name % 'preconditions')
  # save_results(metr_eff, output_dir, dev_dataset_name % 'effects')
  # save_results(metr_conflicts, output_dir, dev_dataset_name % 'conflicts')
  # save_results(metr_stories, output_dir, dev_dataset_name % 'stories')
  # save_results(explanations, output_dir, dev_dataset_name % 'explanations')

  print('\nPARTITION: %s' % p)
  print('Stories:')
  print_dict(metr_stories)
  print('Conflicts:')
  print_dict(metr_conflicts)
  print('Preconditions:')
  print_dict(metr_prec)
  print('Effects:')
  print_dict(metr_eff)


partitions = ['dev', 'test']

for p in partitions:
  consistent_preds = 0
  verifiable_preds = 0
  total = 0
  for expl in explanations:
    if expl['valid_explanation']:
      verifiable_preds += 1
    if expl['story_pred'] == expl['story_label']:
      if len(expl['conflict_pred']) == len(expl['conflict_label']) and expl['conflict_pred'][0] == expl['conflict_label'][0] and expl['conflict_pred'][1] == expl['conflict_label'][1]:
        expl['consistent'] = True
        consistent_preds += 1
      else:
        expl['consistent'] = False
    total += 1
  float(consistent_preds) / total
  print('consistency in {}:{}'.format(p,float(consistent_preds) / total))

