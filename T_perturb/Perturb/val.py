import argparse
import os

import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from datasets import concatenate_datasets, load_from_disk

from T_perturb.Dataloaders.datamodule import CytoMeisterDataModule
from T_perturb.Perturb.trainer import PerturberTrainer
from T_perturb.src.utils import get_idx_for_filtering, read_dataset_files

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

    # Define path to load checkpoint
    n_total_tps = len(tgt_adatas)
    max_seq_length = config['trainer']['max_seq_length'] + 100
    # remove tgt_vocab_size from config
    config['trainer'].pop('tgt_vocab_size')
    config['trainer'].pop('max_seq_length')

    # Initialize model module
    # ----------------------------------------------------------------------------------
    decoder_module = PerturberTrainer(
        condition_dict=condition_dict,
        n_total_tps=n_total_tps,
        tgt_vocab_size=token_no + 50,
        max_seq_length=max_seq_length,
        **config['trainer'],
    )
    data_module = CytoMeisterDataModule(
        n_total_tps=n_total_tps,
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
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
