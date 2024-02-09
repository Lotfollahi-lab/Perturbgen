import datetime
import os
import pickle
import subprocess

import numpy as np
import seaborn as sns
from datasets import load_from_disk
from geneformer import GeneformerPretrainer
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForMaskedLM
from transformers.training_args import TrainingArguments

GPU_NUMBER = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(s) for s in GPU_NUMBER])
os.environ['NCCL_DEBUG'] = 'INFO'
sns.set()


if os.getcwd().split('/')[-3] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/pp'
    )
    print('Changed working directory to root of repository')

tokenized_dir = (
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h.dataset'
)
dataset = load_from_disk(tokenized_dir)
# create an example file of lengths of each example cell and save as pickle
lengths = dataset['length']
lengths_file_path = (
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h_lengths.pkl'
)
with open(lengths_file_path, 'wb') as fp:
    pickle.dump(lengths, fp)

# ---Fine-tune model---


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}


# set model parameters
# max input size
max_input_size = 2**11  # 2048
num_examples = dataset.num_rows
# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 5
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and eval
geneformer_batch_size = 32
# learning schedule
lr_schedule_fn = 'linear'
# warmup steps
warmup_steps = 10_000
# number of epochs
epochs = 3
# optimizer
optimizer = 'adamw'
weight_decay = 0.001
# path to save model
model_output_dir = (
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb/plt/res/Geneformer'
)
# set logging steps

mode = 'masked_lm'  # 'classification'
print(mode)
if mode == 'masked_lm':
    print('masking')
    model = BertForMaskedLM.from_pretrained(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/generative_modelling_omic/Geneformer/',
        output_attentions=False,
        output_hidden_states=False,
    ).to('cuda')
    model.train()

# fine-tune only the classification head
for name, param in model.named_parameters():
    # unfreeze the classification head and last layer of BERT
    if 'cls' in name or 'bert.encoder.layer.5' in name:
        param.requires_grad = True
        print(f'{name} is unfrozen')
    else:
        param.requires_grad = False
        print(f'{name} is frozen')
# define output directory path
current_date = datetime.datetime.now()
datestamp = (
    f'{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}'
)
output_dir = (
    f'./res/Geneformer/{datestamp}_geneformer_CellClassifier_'
    f'L{max_input_size}_B{geneformer_batch_size}_'
    f'LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}'
    f'_E{epochs}_O{optimizer}_F{freeze_layers}_16h/'
)

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, 'pytorch_model.bin')
if os.path.isfile(saved_model_test):
    raise Exception('Model already saved to this directory.')

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training arguments
# training_args = {
#     'learning_rate': max_lr,
#     'do_train': True,
#     'do_eval': True,
#     'evaluation_strategy': 'epoch',
#     'save_strategy': 'epoch',
#     'logging_steps': logging_steps,
#     'group_by_length': True,
#     'length_column_name': 'length',
#     'disable_tqdm': False,
#     'lr_scheduler_type': lr_schedule_fn,
#     'warmup_steps': warmup_steps,
#     'weight_decay': 0.001,
#     'per_device_train_batch_size': geneformer_batch_size,
#     'per_device_eval_batch_size': geneformer_batch_size,
#     'num_train_epochs': epochs,
#     'load_best_model_at_end': True,
#     'output_dir': output_dir,
# }
# define the training arguments
training_args = {
    'learning_rate': max_lr,
    'do_train': True,
    'do_eval': False,
    'group_by_length': True,
    'length_column_name': 'length',
    'disable_tqdm': False,
    'lr_scheduler_type': lr_schedule_fn,
    'warmup_steps': warmup_steps,
    'weight_decay': weight_decay,
    'per_device_train_batch_size': geneformer_batch_size,
    'num_train_epochs': epochs,
    'save_strategy': 'steps',
    'save_steps': np.floor(
        num_examples / geneformer_batch_size / 8
    ),  # 8 saves per epoch
    'logging_steps': 1000,
    'output_dir': model_output_dir,
    'logging_dir': model_output_dir,
}

training_args_init = TrainingArguments(**training_args)

# load gene_ensembl_id:token dictionary
with open(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'generative_modelling_omic/Geneformer/geneformer/token_dictionary.pkl',
    'rb',
) as file:
    token_dictionary = pickle.load(file)


trainer = GeneformerPretrainer(
    model=model,
    args=training_args_init,
    # pretraining corpus
    train_dataset=dataset,
    # file of lengths of each example cell
    example_lengths_file=lengths_file_path,
    token_dictionary=token_dictionary,
)
# train the cell type classifier
trainer.train()
# Save all checkpoints in one folder
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
trainer.save_model(checkpoint_dir)
