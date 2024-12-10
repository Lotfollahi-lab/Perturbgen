import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch

from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.Model.trainer import CountDecoderTrainer
from T_perturb.src.utils import label_encoder
from T_perturb.tests.test_cellgen_training import dummy_dataset
from T_perturb.tests.test_countdecoder_training import dummy_cell_gene_matrix

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/kl11/t_generative/')


tgt_vocab_size = 101  # +1 for padding token
num_samples = 100
num_genes = 100
max_seq_length = 50
n_total_tps = 2
num_samples = 100
batch_size = 4
pred_tps = [1, 2]
context_tps = [1, 2]
d_model = 12

genes_to_perturb = [5, 10]
perturbation_token = 0

src_counts = dummy_cell_gene_matrix(
    num_cells=num_samples,
    num_genes=num_genes,
)
src_dataset = dummy_dataset(
    max_len=max_seq_length,
    vocab_size=tgt_vocab_size,
    num_samples=100,
)
tgt_counts_dict = dummy_cell_gene_matrix(
    num_cells=num_samples,
    num_genes=num_genes,
    total_time_steps=n_total_tps,
)
tgt_datasets = dummy_dataset(
    max_len=max_seq_length,
    vocab_size=tgt_vocab_size,
    num_samples=100,
    total_time_steps=n_total_tps,
)

conditions = None
condition_keys = None
conditions_combined = None

if condition_keys is None:
    condition_keys = 'tmp_batch'
    # create a mock vector if there are no batch effect
    tmp_series = pd.DataFrame(
        {
            condition_keys: np.ones(num_samples),
        }
    )

if isinstance(condition_keys, str):
    condition_keys_ = [condition_keys]
else:
    condition_keys_ = condition_keys

if conditions is None:
    if condition_keys is not None:
        conditions_ = {}
        for cond in condition_keys_:
            conditions_[cond] = tmp_series[cond].unique().tolist()
    else:
        conditions_ = {}
else:
    conditions_ = conditions

if conditions_combined is None:
    if len(condition_keys_) > 1:
        tmp_series['conditions_combined'] = tmp_series[condition_keys].apply(
            lambda x: '_'.join(x), axis=1
        )
    else:
        tmp_series['conditions_combined'] = tmp_series[condition_keys]
    conditions_combined_ = tmp_series['conditions_combined'].unique().tolist()
else:
    conditions_combined_ = conditions_combined

condition_encodings = {
    cond: {k: v for k, v in zip(conditions_[cond], range(len(conditions_[cond])))}
    for cond in conditions_.keys()
}
conditions_combined_encodings = {
    k: v for k, v in zip(conditions_combined_, range(len(conditions_combined_)))
}

tgt_adata_tmp = sc.AnnData(X=tgt_counts_dict['tgt_h5ad_t1'].squeeze(), obs=tmp_series)

if (condition_encodings is not None) and (condition_keys_ is not None):
    conditions = [
        label_encoder(
            tgt_adata_tmp,
            encoder=condition_encodings[condition_keys_[i]],
            condition_key=condition_keys_[i],
        )
        for i in range(len(condition_encodings))
    ]
    conditions = torch.tensor(conditions, dtype=torch.long).T
    conditions_combined = label_encoder(
        tgt_adata_tmp,
        encoder=conditions_combined_encodings,
        condition_key='conditions_combined',
    )
    conditions_combined = torch.tensor(conditions_combined, dtype=torch.long)

decoder_module = CountDecoderTrainer(
    ckpt_masking_path='./T_perturb/T_perturb/tests/'
    'checkpoints/baseline_masking_checkpoint-epoch=00.ckpt',
    ckpt_count_path='./T_perturb/T_perturb/tests/'
    'checkpoints/baseline_counts_checkpoint-epoch=00.ckpt',
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=4,
    num_layers=1,
    d_ff=8,
    max_seq_length=max_seq_length + 10,
    loss_mode='zinb',
    lr=1e-3,
    weight_decay=0.0,
    sequence_length=max_seq_length - 10,
    # lr_scheduler_patience=5.0,
    # lr_scheduler_factor=0.8,
    conditions=conditions_,
    conditions_combined=conditions_combined_,
    n_genes=num_genes,
    dropout=0.0,
    pred_tps=pred_tps,
    context_tps=context_tps,
    n_total_tps=n_total_tps,
    temperature=1.5,
    iterations=19,
    precision='high',
    mask_scheduler='pow',
    output_dir='./T_perturb/T_perturb/tests/res',
    encoder='Transformer_encoder',
    seed=42,
    generate=True,
    var_list=None,
    genes_to_perturb=genes_to_perturb,
    perturbation_token=perturbation_token,
)
data_module = CellGenDataModule(
    src_counts=src_counts,
    tgt_counts_dict=tgt_counts_dict,
    src_dataset=src_dataset,
    tgt_datasets=tgt_datasets,
    batch_size=batch_size,
    num_workers=1,
    pred_tps=pred_tps,
    context_tps=context_tps,
    n_total_tps=n_total_tps,
    train_indices=None,
    test_indices=np.random.choice(100, 20, replace=False),
    max_len=max_seq_length,
    condition_keys=condition_keys_,
    condition_encodings=condition_encodings,
    conditions=conditions,
    conditions_combined=conditions_combined,
)

data_module.setup()

# how can I sample from data_module to get a batch of data?
test_loader = data_module.test_dataloader()

for batch in test_loader:
    print('src', batch['src_input_ids'])
    batch = batch['src_input_ids']
    print(batch)
# how can I perturb the batch replace a token with masked token
perturb_token = [10, 30]
mask_token = 0

trainer = pl.Trainer(
    limit_test_batches=1,  # Limit to a single batch for quick testing
    logger=False,
)
trainer.test(decoder_module, data_module)
