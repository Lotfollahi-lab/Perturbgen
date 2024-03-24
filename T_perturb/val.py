"""Script for validating a classifier on with Pytorch Lightning."""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import scanpy as sc
import torch
import wandb
from datasets import load_from_disk
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import CountDecodertrainer, Petratrainer
from T_perturb.src.utils import (
    label_encoder,
    read_dataset_files,
    stratified_split,
)

RANDOM_SEED = 42


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_mode',
        type=str,
        default='masking',
        help='Mode [masking, count]',
    )
    parser.add_argument(
        '--split',
        type=bool,
        default=False,
        help='split data for extrapolation',
    )
    parser.add_argument(
        '--generate',
        type=bool,
        default=False,
        help='generate data',
    )
    parser.add_argument(
        '--return_embeddings',
        type=bool,
        default=True,
        help='return embedding',
    )
    parser.add_argument(
        '--num_cells',
        type=int,
        default=0,
        help='number of cells to use for testing',
    )
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/Model/checkpoints/'
        '20240322_1802_petra_train_masking_lr_0.001_'
        'wd_0.0_batch_128_mlmp_0.3_tp_1-2-3.ckpt',
        help='path to checkpoint',
    )
    parser.add_argument(
        '--ckpt_count_path',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/Model/checkpoints/'
        '20240322_1900_petra_train_count_lr_0.0005_'
        'wd_0.001_batch_128_zinb_tp_1-2-3.ckpt',
        help='path to checkpoint',
    )
    parser.add_argument(
        '--src_dataset',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset_hvg_src/'
        'cytoimmgen_tokenised_stratified_pairing_0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset_folder',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset_hvg_tgt/',
        help='path to tokenised activated data',
    )

    parser.add_argument(
        '--src_adata',
        type=str,
        default=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
            'T_perturb/pp/res/h5ad_pairing_hvg_src/'
            'cytoimmgen_tokenisation_stratified_pairing_0h.h5ad'
        ),
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
            'T_perturb/pp/res/h5ad_pairing_hvg_tgt'
        ),
        help='path to tgt',
    )
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--mlm_probability', type=float, default=0.5, help='mlm probability'
    )
    parser.add_argument('--max_len', type=int, default=246, help='max sequence length')
    parser.add_argument('--petra_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--count_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--petra_wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--n_workers', type=int, default=64, help='number of workers')
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
    parser.add_argument(
        '--mask_scheduler',
        type=str,
        default='pow',
        help='mask scheduler [cosine, exp, pow]',
    )
    parser.add_argument('--temperature', type=float, default=1.5, help='temperature')
    parser.add_argument('--iterations', type=int, default=19, help='iterations')
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--time_steps',
        type=list,
        default=[1, 2, 3],
        help='time steps to include during training',
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
    tgt_datasets = read_dataset_files(args.tgt_dataset_folder, 'dataset')
    tgt_adatas = read_dataset_files(args.tgt_adata_folder, 'h5ad')
    src_dataset = load_from_disk(args.src_dataset)
    src_adata = sc.read_h5ad(args.src_adata)

    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    tgt_adata_tmp = tgt_adatas['tgt_h5ad_t1'].copy()
    splitting_mode = 'stratified'  # 'random', 'stratified', 'unseen_donor'
    if args.split:
        if splitting_mode == 'stratified':
            # start preprocessing to avoid loading anndata into datamodule
            train_indices, val_indices, test_indices = stratified_split(
                tgt_adata=tgt_adata_tmp,
                train_prop=0.8,  # 0.8,0.1,0.1 train, val, test
                test_prop=0.1,
                groups=['Cell_type', 'Donor'],
                seed=RANDOM_SEED,
            )

            # check that indices are unique to avoid data leakage
            assert len(set(train_indices).intersection(val_indices)) == 0
            assert len(set(train_indices).intersection(test_indices)) == 0
            assert len(set(val_indices).intersection(test_indices)) == 0
        # elif split == 'random':
        #     train, val, test = random_split()
        # elif split == 'unseen_donor':
        #     train, val, test = unseen_donor_split()
        else:
            raise ValueError(
                "split is not available, must be either '"
                "random','stratified' or 'unseen_donor'"
            )
        print(
            f'Number of samples in train set: {len(train_indices)}\n'
            f'Number of samples in val set: {len(val_indices)}\n'
            f'Number of samples in test set: {len(test_indices)}'
        )
    else:
        # return all the indices
        train_indices = list(range(len(src_dataset)))
        val_indices = None
        test_indices = list(range(len(tgt_datasets['tgt_dataset_t1'])))

    if args.loss_mode == 'mse':
        # log normalize data only for mse loss
        sc.pp.normalize_total(src_adata, target_sum=1e4)
        sc.pp.log1p(src_adata)
        for _, tgt_adata in tgt_adatas.items():
            sc.pp.normalize_total(tgt_adata, target_sum=1e4)
            sc.pp.log1p(tgt_adata)
    # ZINB count loss preprocessing
    # ----------------------------------------------------------------------------------
    # TODO: needs to be changed in case batches are different across paired cells
    if isinstance(args.condition_keys, str):
        condition_keys_ = [args.condition_keys]
    else:
        condition_keys_ = args.condition_keys

    if args.conditions is None:
        if args.condition_keys is not None:
            conditions_ = {}
            for cond in condition_keys_:
                conditions_[cond] = tgt_adata_tmp.obs[cond].unique().tolist()
        else:
            conditions_ = {}
    else:
        conditions_ = args.conditions

    if args.conditions_combined is None:
        if len(condition_keys_) > 1:
            tgt_adata_tmp.obs['conditions_combined'] = tgt_adata_tmp.obs[
                args.condition_keys
            ].apply(lambda x: '_'.join(x), axis=1)
        else:
            tgt_adata_tmp.obs['conditions_combined'] = tgt_adata_tmp.obs[
                args.condition_keys
            ]
        conditions_combined_ = (
            tgt_adata_tmp.obs['conditions_combined'].unique().tolist()
        )
    else:
        conditions_combined_ = args.conditions_combined

    condition_encodings = {
        cond: {k: v for k, v in zip(conditions_[cond], range(len(conditions_[cond])))}
        for cond in conditions_.keys()
    }
    conditions_combined_encodings = {
        k: v for k, v in zip(conditions_combined_, range(len(conditions_combined_)))
    }

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
    print('Data loaded and preprocessed.')
    print(tgt_adata_tmp.var['gene_name'])
    # Initialize model module
    # ----------------------------------------------------------------------------------
    if args.test_mode == 'masking':
        pretrained_module = Petratrainer(
            tgt_vocab_size=1820,
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
            return_embeddings=args.return_embeddings,
            generate=args.generate,
            time_steps=args.time_steps,
            gene_names=tgt_adata_tmp.var['gene_name'],
        )
    elif args.test_mode == 'count':
        decoder_module = CountDecodertrainer(
            ckpt_path=args.ckpt_masking_path,
            loss_mode=args.loss_mode,
            lr=args.count_lr,
            weight_decay=args.count_wd,
            lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            conditions=conditions_,
            conditions_combined=conditions_combined_,
            tgt_vocab_size=1820,
            dropout=args.count_dropout,
            generate=args.generate,
            tgt_adata=tgt_adatas,
            time_steps=args.time_steps,
            temperature=args.temperature,
            iterations=args.iterations,
            mask_scheduler=args.mask_scheduler,
        )
    else:
        raise ValueError('test_mode not recognised, needs to be masking or count')

    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    # change dataloader and input
    # create count dictionnary
    tgt_counts_dict = {}
    for keys, tgt_adata in tgt_adatas.items():
        tgt_counts_dict[keys] = tgt_adata.X
    src_counts = src_adata.X

    data_module = PetraDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        src_counts=src_counts,
        tgt_counts_dict=tgt_counts_dict,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=args.shuffle,
        max_len=args.max_len,
        condition_keys=condition_keys_,
        condition_encodings=condition_encodings,
        conditions=conditions,
        conditions_combined=conditions_combined,
        split=args.split,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        time_steps=args.time_steps,
    )

    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_petra')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # The tensorboard logger allows for monitoring the progress of training
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb.init(  # type: ignore
            entity='k-ly',
            project='ttransformer',
            # id=unique_id,  # specify id to log to same run
            group=log_path,  # all runs are saved in one group for multi gpu training
            dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb',
        )
    else:
        wandb.init(  # type: ignore
            entity='k-ly',
            project='ttransformer',
            id=run_id,
            dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb',
        )

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
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        accelerator=accelerator,
        devices=1 if torch.cuda.is_available() else 0,  # infernce only on one gpu
        limit_test_batches=1.0,
    )
    # Finally, kick of the training process.
    if args.test_mode == 'masking':
        trainer.test(
            pretrained_module,
            data_module,
            ckpt_path=args.ckpt_masking_path,
        )
    elif args.test_mode == 'count':
        trainer.test(
            decoder_module,
            data_module,
            args.ckpt_count_path,
        )
    else:
        raise ValueError('test_mode not recognised, needs to be masking or count')


if __name__ == '__main__':
    main()
