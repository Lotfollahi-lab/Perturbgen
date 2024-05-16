"""Script for training a classifier on  with Pytorch Lightning."""
print('imports')
import argparse
import os
import re
from datetime import datetime
import uuid

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from wandb import init  # type: ignore
import gc
from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import CountDecodertrainer, Petratrainer
from T_perturb.src.utils import label_encoder, stratified_split, gears_splitter, randomised_split

print('set up')

# train_dataset = 'cytoimmgen_tokenised_stratified_pairing_16h.dataset'
# use regex to find condition between degs and .dataset

def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_mode',
        type=str,
        default='count',
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
        '--seed',
        type=int,
        default=42,
        help='random seed',
    )
    parser.add_argument(
        '--ckpt_file',
        type=str,
        default='20240306_1831_petra_mode_masking'
        '_lr_0.001_wd_0.0_batch_256_mlmp_0.3_stratified_pairing_16h.ckpt',
        help='path to checkpoint',
    )
    parser.add_argument(
        '--src_dataset_folder',
        type=str,
        default='Projects/2024Mar_Tperturb/'
        'T_perturb/T_perturb/pp/res/dataset_hvg/'
        'cytoimmgen_tokenised_stratified_pairing_0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset_folder',
        type=str,
        default=f'Projects/2024Mar_Tperturb/'
        f'T_perturb/T_perturb/pp/res/dataset_hvg/'
        f'cytoimmgen_tokenised_.dataset',
        help='path to tokenised activated data',
    )

    parser.add_argument(
        '--src_adata_folder',
        type=str,
        default=(
            'Projects/2024Mar_Tperturb/T_perturb/'
            'T_perturb/pp/res/h5ad_pairing_hvg/'
            'cytoimmgen_tokenisation_stratified_pairing_0h.h5ad'
        ),
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default=(
            f'Projects/2024Mar_Tperturb/T_perturb/'
            f'T_perturb/pp/res/h5ad_pairing_hvg/'
            f'cytoimmgen_tokenisation_.h5ad'
        ),
        help='path to tgt',
    )
    
    parser.add_argument(
        '--splitting_mode', type=str, default='random',
        help="data splitting strategy (can be 'random', 'stratified', 'unseen_donor' or 'gears-{gears_modes}')",
    )

    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
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
        default=None,
        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--base_path', type=str, default='/lustre/groups/imm01/workspace/irene.bonafonte', help='home path'
    )
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--d_ff', type=int, default=32, help='d_ff')
    parser.add_argument('--tune_geneformer', type=bool, default=False, help='Whether to tune the geneformer encoder')
    parser.add_argument('--tune_masking', type=bool, default=True, help='Whether to re-train the masking model')
    parser.add_argument('--mse_alpha', type=bool, default=True, help='Weights for mse loss (relative to 0 prediction loss)')
    parser.add_argument('--retrain_masking', type=bool, default=False, help='Whether to retrain from checkpoint masked model')
    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = get_args()
    RANDOM_SEED = args.seed
    dataset_info = re.findall(r'(?<=tokenised_).*(?=.dataset)', args.src_dataset_folder)[0].replace('_control','')

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # Load and preprocess data
    print('Loading and preprocessing data...')
    src_dataset = load_from_disk(args.src_dataset_folder)
    if 'perturbation_embedding' in src_dataset.features.keys():
        d_perturbation_embed = len(src_dataset['perturbation_embedding'][0][0])
    d_perturbation_embed = len(src_dataset['perturbation_embedding'][0][0])
    tgt_dataset = load_from_disk(args.tgt_dataset_folder)
    src_adata = sc.read_h5ad(args.src_adata_folder)
    tgt_adata = sc.read_h5ad(args.tgt_adata_folder)

    if not all(
        tgt_adata.obs['cell_pairing_index'] == tgt_dataset['cell_pairing_index']
    ):
        raise ValueError('Index of adata and tokenized data do not match')

    # Splitting to avoid loading anndata into data module ---------------
    if args.splitting_mode == 'stratified':
        train_indices, val_indices, test_indices = stratified_split(
            tgt_adata=tgt_adata,
            train_prop=0.8,
            test_prop=0.1,
            groups=['Cell_type', 'Donor'],
            seed=RANDOM_SEED,
        )
    elif args.splitting_mode == 'random':
        train_indices, val_indices, test_indices = randomised_split(train_prop=0.8, test_prop=0.1, seed=RANDOM_SEED, adata=tgt_adata)
    elif 'gears' in args.splitting_mode:
        # validation is always 90% of train
        train_indices, val_indices, test_indices = gears_splitter(
            mode=args.splitting_mode.split('-')[1],
            adata=tgt_adata,
            train_prop=0.75, test_prop=0.1, 
            seed=RANDOM_SEED, 
            test_pert_genes=None, test_perts=None,
            base_path=args.base_path,
        )
    else:
        raise ValueError(
            "split is not available, must be either '"
            "random','stratified', 'unseen_donor' or 'gears-*"
        )
    
    # check that indices are unique to avoid data leakage
    assert len(set(train_indices).intersection(val_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(val_indices).intersection(test_indices)) == 0

    print(
        f'Number of samples in train set: {len(train_indices)}\n'
        f'Number of samples in val set: {len(val_indices)}\n'
        f'Number of samples in test set: {len(test_indices)}'
    )

    # Conditions preprocessing for ZINB count loss ---------------------------------
    # create dummy batch variable for when there are no conditions
    if args.condition_keys is None and args.conditions is None and args.conditions_combined is None:
        args.condition_keys = 'dummy'
        tgt_adata.obs[args.condition_keys] = 'dummy_batch'
    
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
                tgt_adata,
                encoder=condition_encodings[condition_keys_[i]],
                condition_key=condition_keys_[i],
            )
            for i in range(len(condition_encodings))
        ]
        conditions = torch.tensor(conditions, dtype=torch.long).T
        conditions_combined = label_encoder(
            tgt_adata,
            encoder=conditions_combined_encodings,
            condition_key='conditions_combined',
        )
        conditions_combined = torch.tensor(conditions_combined, dtype=torch.long)   

    # Pre-process adata and take counts
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
    src_counts = src_adata.X
    tgt_counts = tgt_adata.X
    del src_adata, tgt_adata
    gc.collect()

    print('Data loaded and preprocessed.')
    # Initialize model module
    # ----------------------------------------------------------------------------------
    if args.train_mode == 'masking':
        pretrained_module = Petratrainer(
            tgt_vocab_size=5028,  # 704 for degs, 1819 for tokenised, 5027 for HVG in peturbation assay +1 (padding)
            d_model=256,
            d_encoded_input=512,
            num_heads=8,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_length=2000,
            dropout=args.petra_dropout,
            mlm_probability=args.mlm_probability,
            weight_decay=args.petra_wd,
            lr=args.petra_lr,
            lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            generate=args.generate,
            perturbation_modeling='activation', # activation repression or None (if not perturbation experiment)
            d_perturbation_embed=d_perturbation_embed,
            base_path = args.base_path,
            tune_geneformer=args.tune_geneformer,
        )
    elif args.train_mode == 'count':
        decoder_module = CountDecodertrainer(
            ckpt_path=f'{args.base_path}/Projects/2024Mar_Tperturb/T_perturb/T_perturb/Model/checkpoints/{args.ckpt_file}',
            loss_mode=args.loss_mode,
            lr=args.count_lr,
            weight_decay=args.count_wd,
            lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            conditions=conditions_,
            conditions_combined=conditions_combined_,
            tgt_vocab_size=5028,  # 704 for degs, 1819 for tokenised
            dropout=args.count_dropout,
            d_model=256,
            generate=args.generate,
            perturbation_modeling='activation',
            base_path = args.base_path,
            tune_pretrained=args.tune_masking,
            mse_alpha=args.mse_alpha,
        )

    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')
    # Initialize data module
    # ----------------------------------------------------------------------------------
    if args.train_mode == 'masking':
        data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            src_counts=src_counts,
            tgt_counts=tgt_counts,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            drop_last=False,
            split=args.split,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
    elif args.train_mode == 'count':
        data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            src_counts=src_counts,
            tgt_counts=tgt_counts,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            condition_keys=condition_keys_,
            condition_encodings=condition_encodings,
            conditions=conditions,
            conditions_combined=conditions_combined,
            drop_last=False,
            split=args.split,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
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
            f'mlmp_{args.mlm_probability}_seed{RANDOM_SEED}_{dataset_info}'
        )
        monitor_metric = 'train/perplexity'
        mode = 'min'

    elif args.train_mode == 'count':
        filename = (
            f'{run_id}_mode_{args.train_mode}_lr_{args.count_lr}_wd_{args.count_wd}_'
            f'batch_{args.batch_size}_'
            f'{args.loss_mode}_seed{RANDOM_SEED}_{dataset_info}'
        )
        monitor_metric = 'val/pearson'
        mode = 'max'

    if not args.retrain_masking:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{args.base_path}/Projects/2024Mar_Tperturb/T_perturb/T_perturb/Model/checkpoints',
            filename=filename,
            save_top_k=1,
            verbose=True,
            monitor=monitor_metric,
            mode=mode,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{args.base_path}/Projects/2024Mar_Tperturb/T_perturb/T_perturb/Model/checkpoints',
            filename=args.ckpt_file,
            save_top_k=1,
            verbose=True,
            monitor=monitor_metric,
            mode=mode,
        )

    # The tensorboard logger allows for monitoring the progress of training
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb_logger = WandbLogger(
            project='ttransformer',
            name=f'{run_id}_{str(uuid.uuid4())[:6]}',
            save_dir=log_path,
            log_model='all',
        )  # noqa
    else:
        wandb_logger = WandbLogger(
            project='ttransformer',
            name=f'{run_id}',
            save_dir=log_path,
            log_model='all',
        )  # noqa


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
    # deepspeed_strategy = DeepSpeedStrategy(stage=2)
    deepspeed_strategy = DDPStrategy(find_unused_parameters=True)
    if not args.retrain_masking:   
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
                checkpoint_callback,
                early_stop_callback,
            ],
            max_epochs=args.epochs,
            accelerator='auto',
            devices=-1 if torch.cuda.is_available() else 0,
            strategy=deepspeed_strategy if torch.cuda.device_count() > 1 else 'auto',
        )
    else:
         trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
                checkpoint_callback,
                early_stop_callback,
            ],
            max_epochs=args.epochs,
            accelerator='auto',
            devices=-1 if torch.cuda.is_available() else 0,
            strategy=deepspeed_strategy if torch.cuda.device_count() > 1 else 'auto',
            resume_from_checkpoint=f'{args.base_path}/Projects/2024Mar_Tperturb/T_perturb/T_perturb/Model/checkpoints/{args.ckpt_file}',
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
