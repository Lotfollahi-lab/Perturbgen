import argparse
import os

import pytorch_lightning as pl
import scanpy as sc
import torch
import yaml  # type: ignore
from datasets import concatenate_datasets, load_from_disk

from T_perturb.Dataloaders.datamodule import CytoMeisterDataModule
from T_perturb.Perturb.trainer import PerturberTrainer
from T_perturb.src.utils import (
    condition_for_count_loss,
    get_idx_for_filtering,
    read_dataset_files,
)

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


def main() -> None:
    # Load configuration from a yaml file
    args = get_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load adata
    tgt_adatas = read_dataset_files(config['data']['tgt_adata_folder'], 'h5ad')

    # load dataset
    src_dataset = load_from_disk(config['data']['src_dataset_file'])
    tgt_datasets = read_dataset_files(config['data']['tgt_dataset_folder'], 'dataset')

    if 'loss_mode' in config['trainer']:
        if config['trainer']['loss_mode'] == 'mse':
            for _, tgt_adata in tgt_adatas.items():
                sc.pp.normalize_total(tgt_adata, target_sum=1e4)
                sc.pp.log1p(tgt_adata)

    # change precision for inference to 16-bit
    if config['model']['precision'] == 16:
        device_name = torch.cuda.get_device_name(0)
        precision = (
            'bf16-mixed'
            if 'A100' in device_name or 'H100' in device_name
            else '16-mixed'
        )
        print(f'Using {precision} precision for inference')
    else:
        precision = '32'
        print('Using 32-bit precision for inference')

    token_no = config['trainer']['tgt_vocab_size']
    if 'cond_list' in config['data']:
        full_dataset = concatenate_datasets([src_dataset] + list(tgt_datasets.values()))
        condition_dict = {}
        for condition in config['data']['cond_list']:
            condition_dict[condition] = {
                cell_type: i + token_no
                for i, cell_type in enumerate(full_dataset.unique(condition))
            }
            token_no += len(condition_dict[condition])
    else:
        condition_dict = None

    tgt_adata_tmp = tgt_adatas[f"tgt_h5ad_t{config['trainer']['pred_tps'][0]}"].copy()
    condition_keys = (
        config['data']['condition_keys'] if 'condition_keys' in config['data'] else None
    )
    conditions_combined = (
        config['data']['conditions_combined']
        if 'conditions_combined' in config['data']
        else None
    )
    conditions = (
        config['data']['conditions'] if 'conditions' in config['data'] else None
    )
    (
        conditions,
        condition_encodings,
        conditions_combined,
        conditions_,
        condition_keys_,
        conditions_combined_,
    ) = condition_for_count_loss(
        condition_keys=condition_keys,
        conditions=conditions,
        conditions_combined=conditions_combined,
        tgt_adata_tmp=tgt_adata_tmp,
    )
    # 1. Filter datasets based on condition, if available
    # 2. Extract condition to return gene embeddings
    # ---------------------------------------------------

    # create full dataset to extract metadata for conditioning
    filter_idx = []
    # filter dataset based on condition
    if ('filter_var' in config['data']) and ('filter_cond' in config['data']):
        for dataset in tgt_datasets.values():
            idx_ = get_idx_for_filtering(
                dataset,
                config['data']['filter_cond'],
                config['data']['filter_var'],
            )
            # if len(filter_idx) == 0:
            #     filtered_dataset = None
            # else:
            #     filtered_dataset = dataset.select(idx_)
            filter_idx.extend(idx_)
    if len(filter_idx) > 0:
        # apply condition filter to all datasets
        filter_idx = list(set(filter_idx))
        for i in range(len(tgt_datasets)):
            t = i + 1
            tgt_dataset = tgt_datasets[f'tgt_dataset_t{t}']
            tgt_adata = tgt_adatas[f'tgt_h5ad_t{t}']
            tgt_dataset = tgt_dataset.select(filter_idx)
            tgt_datasets[f'tgt_dataset_t{t}'] = tgt_dataset
            tgt_adata = tgt_adata[filter_idx, :]
            tgt_adatas[f'tgt_h5ad_t{t}'] = tgt_adata
        # for i, dataset in tgt_datasets.items():
        #     tgt_datasets[i] = dataset.select(filter_idx)
        src_dataset = src_dataset.select(filter_idx)

    # Define path to load checkpoint
    n_total_tps = len(tgt_adatas)
    config['trainer']['max_seq_length'] = config['trainer']['max_seq_length'] + 100
    config['trainer']['tgt_vocab_size'] = token_no + 50

    tgt_counts_dict = {}
    for keys, tgt_adata in tgt_adatas.items():
        tgt_counts_dict[keys] = tgt_adata.X
    config['trainer']['n_genes'] = tgt_adata_tmp.shape[1]
    # Initialize model module
    # ----------------------------------------------------------------------------------
    decoder_module = PerturberTrainer(
        condition_dict=condition_dict,
        n_total_tps=n_total_tps,
        conditions=conditions_,
        conditions_combined=conditions_combined_,
        **config['trainer'],
    )
    data_module = CytoMeisterDataModule(
        n_total_tps=n_total_tps,
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        condition_keys=condition_keys_,
        condition_encodings=condition_encodings,
        tgt_counts_dict=tgt_counts_dict,
        conditions=conditions,
        conditions_combined=conditions_combined,
        **config['datamodule'],
    )
    data_module.setup()

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
        devices=1 if torch.cuda.is_available() else 0,  # inference only on one gpu
        precision=precision,
    )
    trainer.test(
        decoder_module, data_module, ckpt_path=config['model']['ckpt_masking_path']
    )


if __name__ == '__main__':
    main()
