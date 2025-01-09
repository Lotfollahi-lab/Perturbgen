import argparse
import os

import pytorch_lightning as pl
import scanpy as sc
import torch
import yaml  # type: ignore
from datasets import load_from_disk

from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.Perturb.trainer import PerturberTrainer
from T_perturb.src.utils import read_dataset_files

# --- 1. Data pre-processing ---
if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/kl11/t_generative/')
print(os.getcwd())
# set seed for reproducibility
seed_no = 42

pl.seed_everything(seed_no)
torch.manual_seed(seed_no)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='T_perturb/T_perturb/configs/eval/HSPC/perturbation.yaml',
    )
    return parser.parse_args()


# Load configuration from a yaml file
args = get_args()
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# load adata
src_adata = sc.read_h5ad(config['data']['src_adata_file'])
tgt_adatas = read_dataset_files(config['data']['tgt_adata_folder'], 'h5ad')

# load dataset
src_dataset = load_from_disk(config['data']['src_dataset_file'])
tgt_datasets = read_dataset_files(config['data']['tgt_dataset_folder'], 'dataset')

# Define path to load checkpoint
n_total_tps = len(tgt_adatas)

# change precision for inference to 16-bit
if config['model']['precision'] == 16:
    device_name = torch.cuda.get_device_name(0)
    precision = (
        'bf16-mixed' if 'A100' in device_name or 'H100' in device_name else '16-mixed'
    )
    print(f'Using {precision} precision for inference')
else:
    precision = '32'
    print('Using 32-bit precision for inference')

decoder_module = PerturberTrainer(n_total_tps=n_total_tps, **config['trainer'])
data_module = CellGenDataModule(
    n_total_tps=n_total_tps,
    src_dataset=src_dataset,
    tgt_datasets=tgt_datasets,
    **config['datamodule'],
)
data_module.setup()
test_loader = data_module.test_dataloader()

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
trainer = pl.Trainer(
    logger=False,
    accelerator=accelerator,
    devices=1 if torch.cuda.is_available() else 0,  # inference only on one gpu
    # limit_test_batches=500.0,
    precision=precision,
)
trainer.test(
    decoder_module, data_module, ckpt_path=config['model']['ckpt_masking_path']
)
