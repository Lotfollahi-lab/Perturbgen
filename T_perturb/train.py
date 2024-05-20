"""Script for training a classifier on with Pytorch Lightning."""

import argparse
import os
import uuid
from datetime import datetime

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import CountDecodertrainer, Petratrainer
from T_perturb.src.utils import (
    label_encoder,
    randomised_split,
    read_dataset_files,
    str2bool,
    stratified_split,
)

RANDOM_SEED = 42

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')


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
        type=str2bool,
        default=True,
        help='split data for extrapolation',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        # default='./T_perturb/T_perturb/plt/res/cytoimmgen',
        default='./T_perturb/T_perturb/plt/res/eb',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        default='random',
        # default='stratified',
        choices=['random', 'stratified', 'unseen_donor'],
        help='splitting mode',
    )
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default='./T_perturb/T_perturb/Model/checkpoints/'
        '20240518_2328_embedding_lr_0.0001'
        '_wd_0.0001_batch_64_mlmp_0.15_tp_1-2-3-epoch=49.ckpt',
        help='path to checkpoint',
    )

    parser.add_argument(
        '--src_dataset',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset',
        # default=(
        #     './T_perturb/T_perturb/pp/res/eb/'
        #     'dataset_all_src/eb_all_Day 00-03.dataset'
        # ),
        # default='./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset_folder',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt',
        # default='./T_perturb/T_perturb/pp/res/eb/dataset_all_tgt',
        # default='./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt',
        help='path to tokenised activated data',
    )
    parser.add_argument(
        '--src_adata',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad',
        # default=(
        #     './T_perturb/T_perturb/pp/'
        #     'res/eb/h5ad_pairing_all_src/eb_all_Day 00-03.h5ad'
        # ),
        # default='./T_perturb/T_perturb/pp/res/cytoimmgen/'
        # 'h5ad_pairing_hvg_src/0h.h5ad',
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt',
        # default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_all_tgt',
        # default='./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt',
        help='path to tgt',
    )
    parser.add_argument(
        '--mapping_dict_path',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl',
        # default='./T_perturb/T_perturb/pp/res/eb/token_id_to_genename_all.pkl'
        default='./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl',
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle')
    parser.add_argument(
        '--epochs', type=int, default=100, help='number of training epochs'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        # default=300,
        # default=2048,
        default=263,
        help='max sequence length',
    )  # check how many genes there are
    parser.add_argument(
        '--tgt_vocab_size',
        type=int,
        # default=1261,
        # default=15280,
        default=2001,
        help='vocab size (max token id + 1) in dataset for padding',
    )
    parser.add_argument('--petra_lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--count_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--petra_wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.01, help='weight decay')
    parser.add_argument(
        '--num_layers', type=int, default=6, help='number of decoder layers'
    )
    parser.add_argument('--d_ff', type=int, default=128, help='feed forward dimension')

    parser.add_argument('--mlm_prob', type=float, default=0.15, help='mlm probability')
    parser.add_argument(
        '--n_workers', type=int, default=32, help='number of workers'
    )  # 64
    parser.add_argument(
        '--loss_mode', type=str, default='zinb', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--petra_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default=None,
        # default='Cell_culture_batch',
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
        # type=list,
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='time steps to include during training',
    )
    parser.add_argument(
        '--var_list',
        # type=list,
        nargs='+',
        type=str,
        # default=['Time_point'],
        default=['Cell_population', 'Cell_type', 'Time_point', 'Donor'],
        help='List of variables to keep in the dataset',
    )
    parser.add_argument(
        '--train_prop',
        type=float,
        default=0.8,
    )
    parser.add_argument(
        '--test_prop',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--mode',
        default='GF_frozen',
        type=str,
        choices=[
            'GF_fine_tuned',
            'GF_frozen',
            'Transformer_encoder',
        ],
        help='mode of encoder',
    )
    args = parser.parse_args()
    return args


def main() -> None:
    # for reproducible results
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    """Run training."""
    args = get_args()
    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # Load and preprocess data
    # ----------------------------------------------------------------------------------
    print('Loading and preprocessing data...')
    tgt_datasets = read_dataset_files(args.tgt_dataset_folder, 'dataset')
    tgt_adatas = read_dataset_files(args.tgt_adata_folder, 'h5ad')
    src_dataset = load_from_disk(args.src_dataset)
    src_adata = sc.read_h5ad(args.src_adata)

    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    tgt_adata_tmp = tgt_adatas[f'tgt_h5ad_t{args.time_steps[0]}']
    if args.split:
        if args.splitting_mode == 'stratified':
            # start preprocessing to avoid loading anndata into datamodule
            train_indices, val_indices, test_indices = stratified_split(
                tgt_adata=tgt_adata_tmp,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                groups=['Cell_type', 'Donor'],
                seed=RANDOM_SEED,
            )

            # check that indices are unique to avoid data leakage
            assert len(set(train_indices).intersection(val_indices)) == 0
            assert len(set(train_indices).intersection(test_indices)) == 0
            assert len(set(val_indices).intersection(test_indices)) == 0
        elif args.splitting_mode == 'random':
            train_indices, val_indices, test_indices = randomised_split(
                adata=tgt_adata_tmp,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                seed=RANDOM_SEED,
            )
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
        test_indices = list(
            range(len(tgt_datasets[f'tgt_dataset_t{args.time_steps[0]}']))
        )
    # check if the train indices are the same for both adata and dataset
    subset_adata = tgt_adata_tmp[train_indices]
    subset_dataset = tgt_datasets[f'tgt_dataset_t{args.time_steps[0]}'].select(
        train_indices
    )
    assert (
        subset_adata.obs['cell_pairing_index'].tolist()
        == subset_dataset['cell_pairing_index']
    )
    # if tgt_adata.X.__class__.__name__ == 'csr_matrix':
    #     tgt_adata.X = tgt_adata.X.A
    # if src_adata.X.__class__.__name__ == 'csr_matrix':
    #     src_adata.X = src_adata.X.A
    if args.loss_mode == 'mse':
        # log normalize data only for mse loss
        sc.pp.normalize_total(src_adata, target_sum=1e4)
        sc.pp.log1p(src_adata)
        for _, tgt_adata in tgt_adatas.items():
            sc.pp.normalize_total(tgt_adata, target_sum=1e4)
            sc.pp.log1p(tgt_adata)

    # ZINB count loss preprocessing
    # ----------------------------------------------------------------------------------

    if args.condition_keys is None:
        args.condition_keys = 'tmp_batch'
        # create a mock vector if there are no batch effect
        tgt_adata_tmp.obs[args.condition_keys] = 1

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
    # count number of unique timepoints
    n_total_timepoints = len(tgt_adatas)
    # Initialize model module
    # ----------------------------------------------------------------------------------
    if args.train_mode == 'masking':
        pretrained_module = Petratrainer(
            # tgt_vocab_size=1820,  # 704 for degs, 1820 for tokenised
            tgt_vocab_size=args.tgt_vocab_size,  # max token id + 1 for padding
            d_model=256,
            num_heads=8,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_length=args.max_len + 100,
            dropout=args.petra_dropout,
            mlm_probability=args.mlm_prob,
            weight_decay=args.petra_wd,
            lr=args.petra_lr,
            # lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            time_steps=args.time_steps,
            total_time_steps=n_total_timepoints,
            mapping_dict_path=args.mapping_dict_path,
            output_dir=args.output_dir,
            mode=args.mode,
        )
    elif args.train_mode == 'count':
        decoder_module = CountDecodertrainer(
            ckpt_masking_path=args.ckpt_masking_path,
            ckpt_count_path=None,
            tgt_vocab_size=args.tgt_vocab_size,
            d_model=256,
            num_heads=8,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_length=args.max_len + 100,
            loss_mode=args.loss_mode,
            lr=args.count_lr,
            weight_decay=args.count_wd,
            # lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            conditions=conditions_,
            conditions_combined=conditions_combined_,
            dropout=args.count_dropout,
            tgt_adata=tgt_adatas,
            time_steps=args.time_steps,
            total_time_steps=n_total_timepoints,
            temperature=args.temperature,
            iterations=args.iterations,
            mask_scheduler=args.mask_scheduler,
            output_dir=args.output_dir,
            mode=args.mode,
        )
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')
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

    if args.train_mode == 'masking':
        data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            src_counts=src_counts,  # TODO: do not pass counts in datamodule
            tgt_counts_dict=tgt_counts_dict,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            split=args.split,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            time_steps=args.time_steps,
            total_time_steps=n_total_timepoints,
            var_list=args.var_list,
        )
    elif args.train_mode == 'count':
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
            total_time_steps=n_total_timepoints,
            var_list=args.var_list,
        )
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_petra')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.
    time_steps_str_ = [str(i) for i in args.time_steps]
    time_steps_str = '-'.join(time_steps_str_)
    if args.train_mode == 'masking':
        filename = (
            f'{run_id}_train_{args.train_mode}_lr_{args.petra_lr}_wd_{args.petra_wd}_'
            f'batch_{args.batch_size}_'
            f'mlmp_{args.mlm_prob}_tp_{time_steps_str}'
        )
        if args.split:
            monitor_metric = 'val/perplexity'
        else:
            monitor_metric = 'train/perplexity'
        mode = 'min'
    elif args.train_mode == 'count':
        filename = (
            f'{run_id}_train_{args.train_mode}_lr_{args.count_lr}_wd_{args.count_wd}_'
            f'batch_{args.batch_size}_'
            f'{args.loss_mode}_tp_{time_steps_str}'
        )
        if args.split:
            monitor_metric = 'val/mse'
            mode = 'min'
        else:
            monitor_metric = 'train/mse'
            mode = 'min'

    checkpoint_path = './T_perturb/T_perturb/Model/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=f'{filename}-' + '{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=50,
        verbose=True,
        # monitor=monitor_metric,
        mode=mode,
    )
    # The tensorboard logger allows for monitoring the progress of training
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb_logger = WandbLogger(
            project='ttransformer',
            name=f'{run_id}_{str(uuid.uuid4())[:6]}',
            save_dir=args.log_dir,
            log_model='all',
        )  # noqa
    else:
        wandb_logger = WandbLogger(
            project='ttransformer',
            name=f'{run_id}',
            save_dir=args.log_dir,
            log_model='all',
        )  # noqa

    # In this simple example we just check if a GPU is available.
    # For training larger models in a distributed settings, this needs more care.

    # Instantiate trainer object.
    # The lightning trainer has a large number of parameters that can improve the
    # training experience. It is recommended to check out the lightning docs for
    # further information.
    # Lightning allows for simple multi-gpu training, gradient accumulation, half
    # precision training, etc. using the trainer class.
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode=mode,
    )
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print('Using device {}.'.format(accelerator))
    # deepspeed_strategy = DeepSpeedStrategy(
    #     stage=2,
    # )
    ddp_strategy = DDPStrategy(find_unused_parameters=True)
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
        strategy=ddp_strategy if torch.cuda.device_count() > 1 else 'auto',
    )
    print('Starting training...')
    if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
        # set working directory to root of repository
        os.chdir(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb/'
        )
        print('Changed working directory to root of repository')

    if args.train_mode == 'masking':
        # Finally, kick of the training process.
        trainer.fit(pretrained_module, data_module)
    elif args.train_mode == 'count':
        trainer.fit(decoder_module, data_module)
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')
    # #collate deepzero checkpoint
    # if torch.cuda.device_count() > 1:
    #     save_path = f'./Model/checkpoints/{filename}'
    #     convert_zero_checkpoint_to_fp32_state_dict(
    #         save_path,
    #         f'{save_path}.pt'
    #     )


if __name__ == '__main__':
    main()
