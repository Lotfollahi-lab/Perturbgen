"""Script for training a classifier on  with Pytorch Lightning."""

import argparse
import os
import re
from datetime import datetime

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import CountDecodertrainer, Petratrainer

# from T_perturb.src.utils import subset_adata_dataset
from wandb import init  # type: ignore

RANDOM_SEED = 42

train_dataset = 'cytoimmgen_tokenised_stratified_pairing_16h.dataset'
# use regex to find condition between degs and .dataset
dataset_info = re.findall(r'(?<=tokenised_).*(?=.dataset)', train_dataset)[0]


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_mode',
        type=str,
        default='masking',
        help='Mode [masking, count]',
    )
    parser.add_argument(
        '--split',
        type=bool,
        default=True,
        help='split data for extrapolation',
    )
    parser.add_argument(
        '--generate',
        type=bool,
        default=False,
        help='generate data',
    )
    parser.add_argument(
        '--num_cells',
        type=int,
        default=0,
        help='number of cells to use for testing',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/Model/checkpoints/'
        '20240228_0113_cora_lr_0.001_wd_0_batch_512_mlm_'
        '0.3_stratified_pairing_16h_mode_masking.ckpt',
        help='path to checkpoint',
    )
    parser.add_argument(
        '--src_dataset',
        type=str,
        default=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            'T_perturb/T_perturb/pp/res/dataset_hvg/'
            'cytoimmgen_tokenised_stratified_pairing_16h.dataset'
        ),
        help='path to tokenised resting data',
    )
    # parser.add_argument(
    #     '--tgt_dataset',
    #     type=str,
    #     default=f'/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    #     f'T_perturb/T_perturb/pp/res/dataset/{train_dataset}',
    #     help='path to tokenised activated data',
    # )
    parser.add_argument(
        '--tgt_dataset_t1',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset_hvg/'
        'cytoimmgen_tokenised_stratified_pairing_16h.dataset',
        help='path to tokenised activated data',
    )
    parser.add_argument(
        '--tgt_dataset_t2',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset_hvg/'
        'cytoimmgen_tokenised_stratified_pairing_40h.dataset',
        help='path to tokenised activated data',
    )
    parser.add_argument(
        '--src_adata_folder',
        type=str,
        default=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            'T_perturb/T_perturb/pp/res/h5ad_pairing_hvg/'
            'cytoimmgen_tokenisation_stratified_pairing_0h.h5ad'
        ),
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default=(
            f'/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
            f'T_perturb/pp/res/h5ad_pairing_hvg/'
            f'cytoimmgen_tokenisation_{dataset_info}.h5ad'
        ),
        help='path to tgt',
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument(
        '--epochs', type=int, default=50, help='number of training epochs'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument('--max_len', type=int, default=246, help='max sequence length')
    parser.add_argument('--petra_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--count_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--petra_wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.001, help='weight decay')
    parser.add_argument(
        '--mlm_probability', type=float, default=0.3, help='mlm probability'
    )
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers')
    parser.add_argument(
        '--loss_mode', type=str, default='zinb', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--petra_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default='Cell_culture_batch',
        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = get_args()

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # Load and preprocess data
    print('Loading and preprocessing data...')
    src_dataset = load_from_disk(args.src_dataset)
    # tgt_dataset = load_from_disk(args.tgt_dataset)
    tgt_dataset_t1 = load_from_disk(args.tgt_dataset_t1)
    tgt_dataset_t2 = load_from_disk(args.tgt_dataset_t2)

    # create dictionary for dataset
    tgt_datasets = {'t1': tgt_dataset_t1, 't2': tgt_dataset_t2}

    src_adata = sc.read_h5ad(args.src_adata_folder)
    tgt_adata = sc.read_h5ad(args.tgt_adata_folder)
    obs_cols = [
        'cell_pairing_index',
        'Cell_type',
        'Donor',
        args.condition_keys,
        'Cell_population',
    ]
    tgt_adata.obs = tgt_adata.obs[obs_cols]
    tgt_adata.var = tgt_adata.var[['gene_name', 'ensembl_id']]
    src_adata.obs = src_adata.obs[obs_cols]
    src_adata.var = src_adata.var[['gene_name', 'ensembl_id']]
    print(src_adata)
    if tgt_adata.X.__class__.__name__ == 'csr_matrix':
        tgt_adata.X = tgt_adata.X.A
    if src_adata.X.__class__.__name__ == 'csr_matrix':
        src_adata.X = src_adata.X.A
    if args.loss_mode == 'mse':
        # log normalize data only for mse loss
        sc.pp.normalize_total(src_adata, target_sum=1e4)
        sc.pp.log1p(src_adata)
        sc.pp.normalize_total(tgt_adata, target_sum=1e4)
        sc.pp.log1p(tgt_adata)
    # if args.num_cells != 0:
    #     src_adata, tgt_adata, src_dataset, tgt_dataset = subset_adata_dataset(
    #         src_adata, tgt_adata, src_dataset,
    #         tgt_dataset, args.num_cells, RANDOM_SEED
    #     )

    if isinstance(args.condition_keys, str):
        condition_keys_ = [args.condition_keys]
    else:
        condition_keys_ = args.condition_keys

    if args.conditions is None:
        if args.condition_keys is not None:
            conditions_ = {}
            for cond in condition_keys_:
                conditions_[cond] = tgt_adata.obs[cond].unique().tolist()
        else:
            conditions_ = {}
    else:
        conditions_ = args.conditions

    if args.conditions_combined is None:
        if len(condition_keys_) > 1:
            tgt_adata.obs['conditions_combined'] = tgt_adata.obs[
                args.condition_keys
            ].apply(lambda x: '_'.join(x), axis=1)
        else:
            tgt_adata.obs['conditions_combined'] = tgt_adata.obs[args.condition_keys]
        conditions_combined_ = tgt_adata.obs['conditions_combined'].unique().tolist()
    else:
        conditions_combined_ = args.conditions_combined
    print('Data loaded and preprocessed.')
    # Initialize model module
    # ----------------------------------------------------------------------------------
    if args.train_mode == 'masking':
        pretrained_module = Petratrainer(
            tgt_vocab_size=1820,  # 704 for degs, 1819 for tokenised
            d_model=256,
            num_heads=8,
            num_layers=1,
            d_ff=32,
            max_seq_length=2000,
            dropout=args.petra_dropout,
            mlm_probability=args.mlm_probability,
            weight_decay=args.petra_wd,
            lr=args.petra_lr,
            lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            batch_size=args.batch_size,
            adata=tgt_adata,
            dataset_info=dataset_info,
            generate=args.generate,
        )
    elif args.train_mode == 'count':
        decoder_module = CountDecodertrainer(
            ckpt_path=args.ckpt_path,
            loss_mode=args.loss_mode,
            lr=args.count_lr,
            weight_decay=args.count_wd,
            lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            conditions=conditions_,
            conditions_combined=conditions_combined_,
            tgt_vocab_size=1820,  # 704 for degs, 1819 for tokenised
            dropout=args.count_dropout,
            d_model=256,
            generate=args.generate,
            tgt_adata=tgt_adata,
        )

        # # Assume `model` is your model
        if args.loss_mode == 'mse':
            condition_encodings = None
            conditions_combined_encodings = None
        else:
            condition_encodings = decoder_module.condition_encodings
            conditions_combined_encodings = decoder_module.conditions_combined_encodings
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')
    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    # change dataloader and input
    if not all(
        tgt_adata.obs['cell_pairing_index'] == tgt_datasets['t1']['cell_pairing_index']
    ):
        raise ValueError('Index of adata and tokenized data do not match')
    if args.train_mode == 'masking':
        data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            src_adata=src_adata,
            tgt_adata=tgt_adata,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            seed=RANDOM_SEED,
            drop_last=False,
            split=args.split,
        )
    elif args.train_mode == 'count':
        data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            src_adata=src_adata,
            tgt_adata=tgt_adata,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            condition_keys=condition_keys_,
            condition_encodings=condition_encodings,
            conditions_combined_encodings=conditions_combined_encodings,
            seed=RANDOM_SEED,
            drop_last=False,
            split=args.split,
        )
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_petra')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.
    if args.train_mode == 'masking':
        filename = (
            f'{run_id}_mode_{args.train_mode}_lr_{args.petra_lr}_wd_{args.petra_wd}_'
            f'batch_{args.batch_size}_'
            f'mlmp_{args.mlm_probability}_{dataset_info}'
        )
        monitor_metric = 'train/perplexity'
        mode = 'min'
    elif args.train_mode == 'count':
        filename = (
            f'{run_id}_mode_{args.train_mode}_lr_{args.count_lr}_wd_{args.count_wd}_'
            f'batch_{args.batch_size}_'
            f'{args.loss_mode}_{dataset_info}'
        )
        monitor_metric = 'val/pearson'
        mode = 'max'
    checkpoint_callback = ModelCheckpoint(
        dirpath='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/Model/checkpoints',
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor=monitor_metric,
        mode=mode,
    )
    # The tensorboard logger allows for monitoring the progress of training
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        init(
            entity='k-ly',
            project='ttransformer',
            # id=unique_id,  # specify id to log to same run
            group=log_path,  # all runs are saved in one group for multi gpu training
            dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb',
        )  # noqa
    else:
        init(
            entity='k-ly',
            project='ttransformer',
            id=run_id,
            dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb',
        )  # noqa

    wandb_logger = WandbLogger(log_model='all')

    # In this simple example we just check if a GPU is available.
    # For training larger models in a distributed settings, this needs more care.
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print('Using device {}.'.format(accelerator))

    # Instantiate trainer object.
    # The lightning trainer has a large number of parameters that can improve the
    # training experience. It is recommended to check out the lightning docs for
    # further information.
    # Lightning allows for simple multi-gpu training, gradient accumulation, half
    # precision training, etc. using the trainer class.
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode=mode,
    )
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            early_stop_callback,
        ],
        max_epochs=args.epochs,
        accelerator=accelerator,
    )

    if args.train_mode == 'masking':
        # Finally, kick of the training process.
        trainer.fit(pretrained_module, data_module)
    elif args.train_mode == 'count':
        trainer.fit(decoder_module, data_module)
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')


if __name__ == '__main__':
    main()
