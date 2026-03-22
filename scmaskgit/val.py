"""Script for training a classifier on with Pytorch Lightning."""

import argparse
import os
import uuid
from datetime import datetime
import pickle

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import load_from_disk, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy  # ,DeepSpeedStrategy

from tests.test_cellgen_training import dummy_dataset
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from scmaskgit.Dataloaders.datamodule import CellGenDataModule
from scmaskgit.Model.trainer import CellGenTrainer, CountDecoderTrainer
from scmaskgit.src.utils import (
    condition_for_count_loss,
    randomised_split,
    read_dataset_files,
    str2bool,
    stratified_split,
)
import wandb

# from pytorch_lightning.utilities.deepspeed import (
#     convert_zero_checkpoint_to_fp32_state_dict,


if os.getcwd().split('/')[-1] != 'scmaskgit':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit')
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
        default=False,
        help='split data for extrapolation',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        default='random',
        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
    )
    parser.add_argument(
        '--split_obs',
        type=str,
        nargs='+',
        default=['celltype_v2'],
    )
    parser.add_argument('--split_value', type=str, default='D351')
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )
    parser.add_argument(
        '--src_dataset',
        type=str,
        default='/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/concatenated_dataset.dataset',
        #     './T_perturb/perturbgen/pp/res/eb/'
        #     'dataset_all_src/eb_all_Day 00-03.dataset'
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--src_adata',
        type=str,
        default='./T_perturb/perturbgen/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad',
        #     './T_perturb/perturbgen/pp/'
        #     'res/eb/h5ad_pairing_all_src/eb_all_Day 00-03.h5ad'
        # 'h5ad_pairing_hvg_src/0h.h5ad',
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default='./T_perturb/perturbgen/pp/res/eb/h5ad_pairing_hvg_tgt',
        help='path to tgt',
    )
    parser.add_argument(
        '--mapping_dict_path',
        type=str,
        default='./T_perturb/perturbgen/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl',
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=str2bool, default=False, help='shuffle')
    parser.add_argument(
        '--epochs', type=int, default=50, help='number of training epochs'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=4096,
        help='max sequence length',
    )  # check how many genes there are
    parser.add_argument(
        '--tgt_vocab_size',
        type=int,
        default=26717,
        help='vocab size (max token id + 1) in dataset for padding',
    )
    parser.add_argument(
        '--cellgen_lr', type=float, default=0.00005, help='learning rate'
    )
    
    parser.add_argument('--count_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--cellgen_wd', type=float, default=0.000001, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.01, help='weight decay')
    parser.add_argument(
        '--num_layers', type=int, default=6, help='number of decoder layers'
    )
    parser.add_argument('--d_ff', type=int, default=96, help='feed forward dimension')

    parser.add_argument('--mlm_prob', type=float, default=0.15, help='mlm probability')
    parser.add_argument(
        '--n_workers', type=int, default=2, help='number of workers'
    )  # 64
    parser.add_argument(
        '--loss_mode', type=str, default='zinb', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--cellgen_dropout', type=float, default=0.03, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default=None,
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
        '--pred_tps',
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='time steps which are predicted',
    )
    parser.add_argument(
        '--context_tps',
        nargs='+',
        type=int,
        default=None,
        help='context time steps in cross-attn',
    )
    parser.add_argument(
        '--var_list',
        nargs='+',
        type=str,
        default=['tissue', 'cell_type'],
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
        '--encoder',
        default='GF_frozen',
        type=str,
        choices=[
            'GF_fine_tuned',
            'GF_frozen',
            'Transformer_encoder',
        ],
        help='mode of encoder',
    )
    parser.add_argument(
        '--pos_encoding_mode',
        type=str,
        default='time_pos_sin',
        choices=['time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'],
        help='positional encoding',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for reproducibility',
    )
    parser.add_argument(
        '--context_mode',
        type=str2bool,
        default=True,
        help='context mode for timepoints',
    )
    args = parser.parse_args()
    return args


def main() -> None:
    # for reproducible results
    """Run training."""
    args = get_args()
    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    # Load and preprocess data
    # ----------------------------------------------------------------------------------
    print('Loading and preprocessing data...')
    src_dataset = []
    # raise
   
    src_adata = None



    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    if args.split:
        if args.splitting_mode == 'stratified':
            # start preprocessing to avoid loading anndata into datamodule
            train_indices, val_indices, test_indices = stratified_split(
                tgt_adata=src_adata,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                groups=args.split_obs,
                seed=args.seed,
            )
            # check that indices are unique to avoid data leakage
            assert len(set(train_indices).intersection(val_indices)) == 0
            assert len(set(train_indices).intersection(test_indices)) == 0
            assert len(set(val_indices).intersection(test_indices)) == 0
        elif args.splitting_mode == 'random':
            train_indices, val_indices, test_indices = randomised_split(
                adata=src_adata,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                seed=args.seed,
            )
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
        raise
    else:
        # return all the indices
        train_indices = list(range(len(src_dataset)))
        val_indices = train_indices
        
        test_indices = list(
            range(492000,558000)
        )
        
        test_dataset = load_from_disk("/lustre/scratch126/cellgen/team361/av13/data/tokenized_data/trace/tokenized_geneformer_multi_organ.dataset")
    
    print('Data loaded and preprocessed.')
  

    # Initialize model module
    # ----------------------------------------------------------------------------------
    trainer_kwargs = {
        'tgt_vocab_size': args.tgt_vocab_size,
        'd_model': 768,
        'num_heads': 8,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_seq_length': args.max_len,
        'mask_scheduler': args.mask_scheduler,
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'encoder': args.encoder,
        'output_dir': args.output_dir,
        'context_mode': args.context_mode,
        'pos_encoding_mode': args.pos_encoding_mode,
    }
    if args.train_mode == 'masking':
        trainer_kwargs['dropout'] = args.cellgen_dropout
        trainer_kwargs['mlm_probability'] = args.mlm_prob
        trainer_kwargs['end_lr'] = args.cellgen_lr
        trainer_kwargs['weight_decay'] = args.cellgen_wd
        trainer_kwargs['mapping_dict_path'] = args.mapping_dict_path
        pretrained_module = CellGenTrainer(**trainer_kwargs)
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')
    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    # change dataloader and input
    # create count dictionnary
    src_counts = None
    # determine global batch size to account for multiple GPUs
    gpu_number = max(torch.cuda.device_count(), 1)
    per_gpu_batch_size = args.batch_size // gpu_number
    data_module_kwargs = {
        'src_dataset': src_dataset,
        'batch_size': per_gpu_batch_size,
        'num_workers': args.n_workers,
        'shuffle': args.shuffle,
        'max_len': args.max_len,
        'split': args.split,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'var_list': args.var_list,
        "test_dataset" :test_dataset
    }
    if args.train_mode == 'masking':
        # TODO: Do not pass src into DataModule
        data_module = CellGenDataModule(**data_module_kwargs)
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_cellgen')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.
    time_steps_str_ = [str(i) for i in args.pred_tps]
    time_steps_str = '-'.join(time_steps_str_)
    if args.train_mode == 'masking':
        filename = (
            f'{run_id}_train_{args.train_mode}_lr_{args.cellgen_lr}'
            f'_wd_{args.cellgen_wd}_batch_{args.batch_size}_'
            f'p{args.pos_encoding_mode}_m_{args.mask_scheduler}'
            f'_tp_{time_steps_str}_s_{args.seed}'
        )
        if val_indices:
            monitor_metric = 'val/perplexity'
        else:
            monitor_metric = 'train/perplexity'
        mode = 'min'
   

    checkpoint_path = os.path.join(args.output_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=f'{filename}-' + '{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
        monitor="train/perplexity",
        mode=mode,
    )
    # The tensorboard logger allows for monitoring the progress of training
    # Configure WandbLogger with unique name for each run
    run_name = (
        f'{run_id}_{str(uuid.uuid4())[:6]}' if torch.cuda.device_count() > 1 else run_id
    )

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
    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            # early_stop_callback,
            checkpoint_callback,
        ],
        max_epochs=args.epochs,
        accelerator=accelerator,
        precision='bf16-mixed',
        devices=-1 if torch.cuda.is_available() else 0,
        strategy=ddp_strategy if torch.cuda.device_count() > 1 else 'auto',
    )
    print('Starting training...')


    trainer.test(pretrained_module, data_module, 
    ckpt_path = "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output2/checkpoints/20250110_2325_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=02.ckpt"
    )

    


if __name__ == '__main__':
    main()
