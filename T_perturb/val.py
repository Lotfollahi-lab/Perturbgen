"""Script for validating a classifier on with Pytorch Lightning."""

import argparse
import os
import uuid
import warnings
from datetime import datetime

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import concatenate_datasets, load_from_disk
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy  # DeepSpeedStrategy

from T_perturb.Dataloaders.datamodule import CytoMeisterDataModule
from T_perturb.Model.trainer import CountDecoderTrainer, CytoMeisterTrainer
from T_perturb.src.utils import (
    condition_for_count_loss,
    get_idx_for_filtering,
    randomised_split,
    read_dataset_files,
    str2bool,
    stratified_split,
)

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/kl11/t_generative')
    print('Changed working directory to root of repository')


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_mode',
        type=str,
        default='count',
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
        default='./T_perturb/T_perturb/plt/res/cytoimmgen',
        # default='./T_perturb/T_perturb/plt/res/eb',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        # default='random',
        default='stratified',
        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
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
        '--split_obs',
        type=str,
        nargs='+',
        default=None,
        # default=['celltype_v2'],
    )
    parser.add_argument('--split_value', type=str, default='D351')
    parser.add_argument('--use_positional_encoding', type=str2bool, default=False)
    parser.add_argument('--layer_norm', type=str2bool, default=False)
    parser.add_argument('--add_cell_time', type=str2bool, default=False)

    parser.add_argument(
        '--d_condc',
        type=int,
        default=None,
        help='One Hot dimension',
    )
    parser.add_argument(
        '--d_condt',
        type=int,
        default=768,
        help='One Hot dimension',
    )
    parser.add_argument(
        '--generate',
        type=str2bool,
        default=True,
        help='generate data',
    )
    parser.add_argument(
        '--return_embeddings',
        type=str2bool,
        default=False,
        help='return embedding',
    )
    parser.add_argument(
        '--return_attn',
        type=str2bool,
        default=False,
        help='return attention',
    )
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )
    parser.add_argument(
        '--ckpt_count_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )
    parser.add_argument(
        '--mapping_dict_path',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl',
        # default='./T_perturb/T_perturb/pp/res/eb/token_id_to_genename_all.pkl'
        default='./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl',
    )
    parser.add_argument(
        '--src_dataset',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset',
        # default=(
        #     './T_perturb/T_perturb/pp/res/eb/'
        #     'dataset_all_src/eb_all_Day 00-03.dataset'
        # ),
        default='./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset_folder',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt',
        default='./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt/',
        help='path to tokenised activated data',
    )

    parser.add_argument(
        '--src_adata',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad',
        default=(
            './T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad'
        ),
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt',
        default=('./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt'),
        help='path to tgt',
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('--num_node', type=int, default=1)
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=300,
        # default=2048,
        # default=263,
        help='max sequence length',
    )
    parser.add_argument(
        '--tgt_vocab_size',
        type=int,
        # default=1261,
        # default=15280,
        default=1997,
        help='vocab size (max token id + 1) in dataset for padding',
    )
    parser.add_argument(
        '--cellgen_lr', type=float, default=0.0001, help='learning rate'
    )
    parser.add_argument('--count_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--cellgen_wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--n_workers', type=int, default=32, help='number of workers')
    parser.add_argument(
        '--num_layers', type=int, default=6, help='number of decoder layers'
    )
    parser.add_argument('--d_ff', type=int, default=128, help='feed forward dimension')
    parser.add_argument(
        '--loss_mode', type=str, default='mse', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--cellgen_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        # default='Cell_culture_batch',
        default=None,
        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument(
        '--mask_scheduler',
        type=str,
        default='cosine',
        help='mask scheduler [cosine, exp, pow]',
    )
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument('--sequence_length', type=int, default=150, help='iterations')
    parser.add_argument('--iterations', type=int, default=20, help='iterations')
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--pred_tps',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='time steps to include during training',
    )
    parser.add_argument(
        '--context_tps',
        type=int,
        nargs='+',
        default=None,
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
        '--cond_list',
        nargs='+',
        type=str,
        help='List of variables to form condition tokens',
    )
    parser.add_argument(
        '--encoder',
        default='GF_frozen',
        type=str,
        choices=[
            'GF_fine_tuned',
            'GF_frozen',
            'Transformer_encoder',
            'scmaskgit',
        ],
        help='mode of encoder',
    )
    parser.add_argument(
        '--encoder_path',
        type=str,
        default=None,
        help='path to pre-trained encoder',
    )
    parser.add_argument(
        '--tokenid_to_rowid_path',
        type=str,
        default=None,
        help='path to tokenid to rowid mapping file',
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
    parser.add_argument(
        '--pos_encoding_mode',
        type=str,
        default='time_pos_sin',
        help='positional encoding',
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='embedding dimension',
    )
    parser.add_argument(
        '--deg_pkl_path',
        type=str,
        default=None,
        help='path to deg pkl file',
    )
    parser.add_argument(
        '--return_gene_embs',
        type=str2bool,
        default=False,
        help='return gene embeddings',
    )
    parser.add_argument(
        '--gene_embs_condition',
        type=str,
        default=None,
        help='aggregate gene embeddings over condition',
    )
    parser.add_argument(
        '--filter_cond',
        type=str,
        nargs='+',
        default=None,
        help='condition to filter tgt datasets',
    )
    parser.add_argument(
        '--filter_var',
        type=str,
        default=None,
        help='covariate to filter tgt datasets',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=3,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = get_args()
    print('positional encoding:', args.pos_encoding_mode)

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    # Load and preprocess data
    print('Loading and preprocessing data...')
    tgt_datasets = read_dataset_files(
        args.tgt_dataset_folder,
        'dataset',
    )
    tgt_adatas = read_dataset_files(
        args.tgt_adata_folder,
        'h5ad',
    )
    src_dataset = load_from_disk(args.src_dataset)
    src_adata = sc.read_h5ad(args.src_adata)

    if args.loss_mode == 'mse':
        # log normalize data only for mse loss
        sc.pp.normalize_total(src_adata, target_sum=1e4)
        sc.pp.log1p(src_adata)
        for _, tgt_adata in tgt_adatas.items():
            sc.pp.normalize_total(tgt_adata, target_sum=1e4)
            sc.pp.log1p(tgt_adata)

    # Classifier-free guidance pre-processing
    # ---------------------------------------
    # create dictionnary of metadata for classifier-free guidance
    # needs to be completed before any filtering to initialize
    # all condition tokens
    token_no = args.tgt_vocab_size
    if args.cond_list is not None:
        full_dataset = concatenate_datasets([src_dataset] + list(tgt_datasets.values()))
        condition_dict = {}
        for condition in args.cond_list:
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
    if (args.filter_var is not None) and (args.filter_cond is not None):
        for dataset in tgt_datasets.values():
            idx_ = get_idx_for_filtering(
                dataset,
                args.filter_cond,
                args.filter_var,
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
        src_adata = src_adata[filter_idx, :]
        if args.gene_embs_condition is not None:
            # filter for keys which are in pred_tps
            pred_dataset = {
                key: tgt_datasets[key]
                for key in tgt_datasets
                if key in {f'tgt_dataset_t{tp}' for tp in args.pred_tps}
            }
            all_pred_dataset = concatenate_datasets(list(pred_dataset.values()))
            # check if filter_cond is the same as all_pred_dataset
            all_pred_condition = all_pred_dataset.unique(args.filter_var)
            if all_pred_condition != args.filter_cond:
                warnings.warn(
                    f'Filtering gene embs for {all_pred_condition} '
                    f'instead of only {args.filter_cond} '
                    f'if this is not the intended behaviour, please select the '
                    f'corresponding pred_tps for the gene embs condition '
                )
            gene_embs_list = all_pred_dataset.unique(args.gene_embs_condition)
            print(
                f'Return gene embs for {gene_embs_list} '
                f'in condition {args.gene_embs_condition} '
                f'filtered for {all_pred_condition}.'
            )
        else:
            gene_embs_list = None
    else:
        if args.gene_embs_condition is not None:
            full_tgt_dataset = concatenate_datasets(list(tgt_datasets.values()))
            gene_embs_list = full_tgt_dataset.unique(args.gene_embs_condition)
            print(
                f'Return gene embs for {gene_embs_list} '
                f'in {args.gene_embs_condition}.'
            )
        else:
            gene_embs_list = None

    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    tgt_adata_tmp = tgt_adatas[f'tgt_h5ad_t{args.pred_tps[0]}'].copy()

    # ZINB and NB count loss preprocessing
    # ------------------------------------

    (
        conditions,
        condition_encodings,
        conditions_combined,
        conditions_,
        condition_keys_,
        conditions_combined_,
    ) = condition_for_count_loss(
        args.condition_keys, args.conditions, args.conditions_combined, tgt_adata_tmp
    )

    print('Data loaded and preprocessed.')
    # Preparing train-test split
    # --------------------------
    if args.split:
        if args.splitting_mode == 'stratified':
            # start preprocessing to avoid loading anndata into datamodule
            train_indices, val_indices, test_indices = stratified_split(
                tgt_adata=tgt_adata_tmp,
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
                adata=tgt_adata_tmp,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                seed=args.seed,
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
            range(len(tgt_datasets[f'tgt_dataset_t{args.pred_tps[0]}']))
        )
    # check if the train indices are the same for both adata and dataset
    subset_adata = tgt_adata_tmp[train_indices]
    subset_dataset = tgt_datasets[f'tgt_dataset_t{args.pred_tps[0]}'].select(
        train_indices
    )
    adata_idx = subset_adata.obs['index'].tolist()
    dataset_idx = list(map(str, subset_dataset['cell_pairing_index']))
    assert adata_idx == dataset_idx, (
        'Cell pairing indices do not match ' 'between AnnData and Dataset objects'
    )
    # count number of unique timepoints
    n_total_tps = len(tgt_adatas)
    # Initialize model module
    # ----------------------------------------------------------------------------------
    test_kwargs = {
        'tgt_vocab_size': token_no + 50,  # add 50 for extra tokens
        'd_model': args.d_model,
        'num_heads': 8,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_seq_length': args.max_len + 100,
        'dropout': 0,
        'generate': args.generate,
        'context_tps': args.context_tps,
        'pred_tps': args.pred_tps,
        'n_total_tps': n_total_tps,
        'mask_scheduler': args.mask_scheduler,
        'pos_encoding_mode': args.pos_encoding_mode,
        'output_dir': args.output_dir,
        'encoder': args.encoder,
        'var_list': args.var_list,
        'encoder_path': args.encoder_path,
        'condition_dict': condition_dict,
        'temperature': args.temperature,
        'iterations': args.iterations,
        'sequence_length': args.sequence_length,
        'mapping_dict_path': args.mapping_dict_path,
    }
    if args.test_mode == 'masking':
        test_kwargs['weight_decay'] = args.cellgen_wd
        test_kwargs['end_lr'] = args.cellgen_lr
        test_kwargs['return_embeddings'] = args.return_embeddings
        test_kwargs['return_gene_embs'] = args.return_gene_embs
        test_kwargs['gene_names'] = tgt_adata_tmp.var['gene_name']
        test_kwargs['context_mode'] = args.context_mode
        test_kwargs['return_attn'] = args.return_attn
        test_kwargs['tokenid_to_rowid_path'] = args.tokenid_to_rowid_path
        test_kwargs['deg_pkl_path'] = args.deg_pkl_path
        test_kwargs['gene_embs_list'] = gene_embs_list
        test_kwargs['gene_embs_condition'] = args.gene_embs_condition
        pretrained_module = CytoMeisterTrainer(**test_kwargs)

    elif args.test_mode == 'count':
        test_kwargs['ckpt_masking_path'] = args.ckpt_masking_path
        test_kwargs['ckpt_count_path'] = args.ckpt_count_path
        test_kwargs['loss_mode'] = args.loss_mode
        test_kwargs['d_condc'] = args.d_condc
        test_kwargs['d_condt'] = args.d_condt
        test_kwargs['layer_norm'] = args.layer_norm
        test_kwargs['dropout'] = args.count_dropout
        test_kwargs['use_positional_encoding'] = args.use_positional_encoding
        test_kwargs['add_cell_time'] = args.add_cell_time
        test_kwargs['weight_decay'] = args.count_wd
        test_kwargs['lr'] = args.count_lr
        test_kwargs['conditions'] = conditions_
        test_kwargs['conditions_combined'] = conditions_combined_
        test_kwargs['tgt_adata'] = tgt_adatas
        test_kwargs['n_samples'] = args.n_samples
        test_kwargs['seed'] = args.seed
        test_kwargs['n_genes'] = src_adata.shape[1]
        decoder_module = CountDecoderTrainer(**test_kwargs)
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
    data_module_kwargs = {
        'src_dataset': src_dataset,
        'tgt_datasets': tgt_datasets,
        'batch_size': args.batch_size,
        'num_workers': args.n_workers,
        'shuffle': args.shuffle,
        'max_len': args.max_len,
        'split': args.split,
        'src_counts': src_counts,
        'tgt_counts_dict': tgt_counts_dict,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'n_total_tps': n_total_tps,
        'var_list': args.var_list,
        'condition_keys': condition_keys_,
        'condition_encodings': condition_encodings,
        'conditions': conditions,
        'conditions_combined': conditions_combined,
    }

    data_module = CytoMeisterDataModule(
        **data_module_kwargs,
    )
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_CytoMeister')
    log_path = os.path.join(
        './T_perturb/T_perturb/wandb/wandb',
        run_id,
    )
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # The tensorboard logger allows for monitoring the progress of training
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb_logger = WandbLogger(
            project='ttransformer',
            name=f'{run_id}_{str(uuid.uuid4())[:6]}',
            save_dir='./T_perturb/T_perturb/wandb/wandb',
            log_model=True,
        )  # noqa
    else:
        wandb_logger = WandbLogger(
            project='ttransformer',
            name=f'{run_id}',
            save_dir='./T_perturb/T_perturb/wandb/wandb',
            log_model=True,
        )

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

    # deepspeed_strategy = DeepSpeedStrategy(
    #     stage=2,
    # )
    # if torch.cuda.is_available():
    #     cuda_device_name = torch.cuda.get_device_name()
    # if ('A100' in cuda_device_name) or ('NVIDIA H100 80GB HBM' in cuda_device_name):
    #     print(f'Using {cuda_device_name} for training')
    #     precision = 'bf16-mixed'
    # else:
    #     precision = '16-mixed'
    ddp_strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        accelerator=accelerator,
        num_nodes=args.num_node,
        devices=-1 if torch.cuda.is_available() else 0,  # inference only on one gpu
        strategy=ddp_strategy if torch.cuda.device_count() > 1 else 'auto',
        limit_test_batches=5,
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
        )
    else:
        raise ValueError('test_mode not recognised, needs to be masking or count')


if __name__ == '__main__':
    main()
