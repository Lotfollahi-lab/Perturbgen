import datetime
import os
import pickle
import subprocess

#  imports
from collections import Counter

import seaborn as sns
from datasets import load_from_disk
from geneformer import DataCollatorForCellClassification
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer
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

tokenized_dir = './res/dataset/cytoimmgen_degs_random_pairing_16h.dataset'
dataset = load_from_disk(tokenized_dir)

# per scDeepsort published method, drop cell types representing <0.5% of cells
celltype_counter = Counter(dataset['Cell_type'])
total_cells = sum(celltype_counter.values())
cells_to_keep = [k for k, v in celltype_counter.items() if v > (0.005 * total_cells)]


def if_not_rare_celltype(example):
    return example['Cell_type'] in cells_to_keep


dataset_subset = dataset.filter(if_not_rare_celltype, num_proc=16)

# shuffle datasets and rename columns
dataset_shuffled = dataset_subset.shuffle(seed=42)
dataset_shuffled = dataset_shuffled.rename_column('Cell_type', 'label')

# create dictionary of cell types : label ids
target_names = list(Counter(dataset_shuffled['label']).keys())
target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))


# change labels to numerical ids
def classes_to_ids(example):
    example['label'] = target_name_id_dict[example['label']]
    return example


labeled_trainset = dataset_shuffled.map(classes_to_ids, num_proc=16)

# create 80/20 train/eval splits
labeled_train_split = labeled_trainset.select(
    [i for i in range(0, round(len(labeled_trainset) * 0.8))]
)
labeled_eval_split = labeled_trainset.select(
    [i for i in range(round(len(labeled_trainset) * 0.8), len(labeled_trainset))]
)

# filter dataset for cell types in corresponding training set
trained_labels = list(Counter(labeled_train_split['label']).keys())


def if_trained_label(example):
    return example['label'] in trained_labels


labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

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
warmup_steps = 500
# number of epochs
epochs = 4
# optimizer
optimizer = 'adamw'

# set logging steps
logging_steps = round(len(labeled_train_split) / geneformer_batch_size / 10)

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/generative_modelling_omic/Geneformer/',
    num_labels=len(target_name_id_dict.keys()),
    output_attentions=False,
    output_hidden_states=False,
).to('cuda')

# define output directory path
current_date = datetime.datetime.now()
datestamp = (
    f'{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}'
)
output_dir = (
    f'./res/Geneformer/{datestamp}_geneformer_CellClassifier_'
    f'L{max_input_size}_B{geneformer_batch_size}_'
    f'LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}'
    f'_E{epochs}_O{optimizer}_F{freeze_layers}/'
)

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, 'pytorch_model.bin')
if os.path.isfile(saved_model_test):
    raise Exception('Model already saved to this directory.')

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training arguments
training_args = {
    'learning_rate': max_lr,
    'do_train': True,
    'do_eval': True,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'logging_steps': logging_steps,
    'group_by_length': True,
    'length_column_name': 'length',
    'disable_tqdm': False,
    'lr_scheduler_type': lr_schedule_fn,
    'warmup_steps': warmup_steps,
    'weight_decay': 0.001,
    'per_device_train_batch_size': geneformer_batch_size,
    'per_device_eval_batch_size': geneformer_batch_size,
    'num_train_epochs': epochs,
    'load_best_model_at_end': True,
    'output_dir': output_dir,
}

training_args_init = TrainingArguments(**training_args)

# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=labeled_train_split,
    eval_dataset=labeled_eval_split_subset,
    compute_metrics=compute_metrics,
)
# train the cell type classifier
trainer.train()
predictions = trainer.predict(labeled_eval_split_subset)
with open(f'{output_dir}predictions.pickle', 'wb') as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics('eval', predictions.metrics)
trainer.save_model(output_dir)
