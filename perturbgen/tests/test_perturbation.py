# %%
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch

from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.Perturb.trainer import PerturberTrainer

from perturbgen.src.utils import label_encoder
from perturbgen.tests.utils import dummy_dataset, dummy_cell_gene_matrix

# %%
seed_no = 42
pl.seed_everything(seed_no)
torch.manual_seed(seed_no)

# %%
if os.getcwd().split('/')[-1] != 'T_perturb':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/')

# %%
tgt_vocab_size = 100  # +1 for padding token
num_samples = 100
num_genes = 100
max_seq_length = 50
n_total_tps = 2
num_samples = 100
batch_size = 4
pred_tps = [1, 2]
context_tps = [1, 2]
d_model = 768

genes_to_perturb = ['ISG15']
perturbation_token = 1


# %%
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

# %%
# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
trainer_params = {
    'tgt_vocab_size': tgt_vocab_size,
    'd_model': d_model,
    'num_heads': 4,
    'num_layers': 1,
    'd_ff': 8,
    'max_seq_length': max_seq_length + 10,
    'dropout': 0.0,
    'pred_tps': pred_tps,
    'context_tps': context_tps,
    'n_total_tps': n_total_tps,
    'precision': 'high',
    'mask_scheduler': 'pow',
    'mapping_dict_path': '/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/token_id_to_genename_5000_hvg.pkl',
    'output_dir': './T_perturb/perturbgen/tests/res',
    'encoder': 'scmaskgit',
    'encoder_path': '/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt',
    'var_list': None,
    'genes_to_perturb': genes_to_perturb,
    'perturbation_mode': 'overexpress',
    'perturbation_sequence': 'tgt',
    'validation_mode': 'inference',
    'context_mode': True,
    'temperature':1.5,
    'iterations': 19,
    'sequence_length': max_seq_length - 10,
    'pos_encoding_mode': 'time_pos_sin',
    'mask_scheduler': 'cosine',
    'var_list': ['cell_type'],
    'condition_dict': None
}
decoder_module = PerturberTrainer(
    **trainer_params
)

# %%

data_module = PerturbGenDataModule(
    batch_size=batch_size,
    src_counts=src_counts,
    tgt_counts_dict=tgt_counts_dict,
    src_dataset=src_dataset,
    tgt_datasets=tgt_datasets,
    num_workers=1,
    pred_tps=pred_tps,
    context_tps=context_tps,
    n_total_tps=n_total_tps,
    split=False,
    max_len=max_seq_length,
    var_list=['cell_type'],
    use_weighted_sampler=False
)

# %%
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
trainer = pl.Trainer(
    limit_test_batches=1,  # Limit to a single batch for quick testing
    logger=False,
    accelerator=accelerator,
    devices=1,  # inference only on one gpu
    precision=16 if accelerator == 'gpu' else 32,  # use mixed precision on gpu for faster inference
    enable_progress_bar=False,
    enable_model_summary=False,
    num_sanity_val_steps=0,
)
trainer.test(
    decoder_module, 
    data_module,
    )