"""Script for training a classifier on with Pytorch Lightning."""

import argparse
import os
import pickle
import uuid
from datetime import datetime

import pytorch_lightning as pl

import scanpy as sc
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.Model.trainer import CellGenTrainer, CountDecoderTrainer
from T_perturb.src.utils import read_dataset_files

# from T_perturb.Model.trainers import CountDecoderTrainer
# label_encoder,; randomised_mapping_dir_split,;;
# randomised_split,; read_dataset_files,
from T_perturb.src.utils import dataset_split, str2bool

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/chang/CellGen/')
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
    '--mapping_dict_path',
    type=str,
    # default='./T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl',
    # default='./T_perturb/pp/res/eb/token_id_to_genename_all.pkl'
    default='/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl',
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        # default='./T_perturb/plt/res/cytoimmgen',
        default='./T_perturb/plt/res/norman',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        default='random',
        # default='stratified',
        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
    )
    parser.add_argument('--split_obs', type=str, default='Donor')
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
        default='./T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset',
        # default=(
        #     './T_perturb/pp/res/eb/'
        #     'dataset_all_src/eb_all_Day 00-03.dataset'
        # ),
        # default='./T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset',
        type=str,
        default='./T_perturb/pp/res/eb/dataset_hvg_tgt',
        # default='./T_perturb/pp/res/eb/dataset_all_tgt',
        # default='./T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt',
        help='path to tokenised activated data',
    )
    parser.add_argument(
        '--src_adata',
        type=str,
        default='./T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad',
        # default=(
        #     './T_perturb/pp/'
        #     'res/eb/h5ad_pairing_all_src/eb_all_Day 00-03.h5ad'
        # ),
        # default='./T_perturb/pp/res/cytoimmgen/'
        # 'h5ad_pairing_hvg_src/0h.h5ad',
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata',
        type=str,
        default='./T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt',
        # default='./T_perturb/pp/res/eb/h5ad_pairing_all_tgt',
        # default='./T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt',
        help='path to tgt file',
    )
    parser.add_argument(
        '--tgt_adata_path',
        type=str,
        default='./T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt',
        # default='./T_perturb/pp/res/eb/h5ad_pairing_all_tgt',
        # default='./T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt',
        help='path to tgt',
    )

    parser.add_argument(
        '--pairing_metadata',
        type=str,
        default=None,
        help='path to pkl file used for cell pairing',
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
    parser.add_argument(
        '--cellgen_lr', type=float, default=0.0001, help='learning rate'
    )
    parser.add_argument('--count_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--cellgen_wd', type=float, default=0.0001, help='weight decay')
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
    parser.add_argument('--cellgen_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default=None,
        # default='Cell_culture_batch',
        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--n_task_conditions',
        # type=list,
        type=int,
        default=1,
        help='Number of task tokens corresponds to number of MoE classes',
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
        '--encoder_type',
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
        '--moe_type',
        default='none',
        type=str,
        choices=[
            'none',
            'moe_attention',
            'moe_ffn',
        ],
        help='mode of encoder',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for reproducibility',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='alpha for multi-task learning',
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
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    # Load and preprocess data
    # ----------------------------------------------------------------------------------

    with open(args.mapping_dict_path, 'rb') as f:
        tokenid_to_genename_dict = pickle.load(f)
    
    print('Loading and preprocessing data...')

    tgt_dataset = load_from_disk(args.tgt_dataset)
    src_dataset = load_from_disk(args.src_dataset)
    # with open(args.pairing_metadata, 'rb') as f:
    #     pairing_metadata = pickle.load(f)
    # Preprocessing adata for cell pairing
    if args.train_mode == 'masking':
        
        tgt_adatas = read_dataset_files(args.tgt_adata_path, 'h5ad')
        
        tgt_counts_dict = {}
        for keys, tgt_adata in tgt_adatas.items():
            tgt_counts_dict[keys] = tgt_adata.X
            
        print(f'tgt_counts_dict file shape: {len(tgt_counts_dict)}')
        print("tgt_counts_dict with keys() :", len(tgt_counts_dict.keys()))
        print("tgt_counts_dict with values():", len(tgt_counts_dict.values()))

        tgt_adata = sc.read_h5ad(args.tgt_adata)

        src_adata = sc.read_h5ad(args.src_adata)
        src_counts = src_adata.X

        
    if args.train_mode == 'count':
        tgt_adatas = read_dataset_files(args.tgt_adata_path, 'h5ad')
        # only load counts not entire anndata
        tgt_counts_dict = {}
        for keys, tgt_adata in tgt_adatas.items():
            tgt_counts_dict[keys] = tgt_adata.X
        src_adata = sc.read_h5ad(args.src_adata)
        src_counts = src_adata.X

        if args.loss_mode == 'mse':
            # log normalize data only for mse loss
            sc.pp.normalize_total(src_adata, target_sum=1e4)
            sc.pp.log1p(src_adata)
            for _, tgt_adata in tgt_adatas.items():
                sc.pp.normalize_total(tgt_adata, target_sum=1e4)
                sc.pp.log1p(tgt_adata)
            # ZINB count loss preprocessing
        # --------------------------------------------------
        print("Keys in tgt_adatas:", tgt_adatas.keys())
        tgt_adata_tmp = tgt_adatas[f'tgt_h5ad_tp']
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
            cond: {
                k: v for k, v in zip(conditions_[cond], range(len(conditions_[cond])))
            }
            for cond in conditions_.keys()
        }
        conditions_combined_encodings = {
            k: v for k, v in zip(
                conditions_combined_,
                range(len(conditions_combined_))
                )
        }

        # if (condition_encodings is not None) and (condition_keys_ is not None):
        #     conditions = [
        #         label_encoder(
        #             tgt_adata_tmp,
        #             encoder=condition_encodings[condition_keys_[i]],
        #             condition_key=condition_keys_[i],
        #         )
        #         for i in range(len(condition_encodings))
        #     ]
        #     conditions = torch.tensor(conditions, dtype=torch.long).T
        #     conditions_combined = label_encoder(
        #         tgt_adata_tmp,
        #         encoder=conditions_combined_encodings,
        #         condition_key='conditions_combined',
        #     )
        #     conditions_combined = torch.tensor(conditions_combined, dtype=torch.long)
        # else:
        #     src_counts = None
        #     tgt_counts_dict = None
        #     tgt_adatas = None
        #     conditions = None
        #     conditions_combined = None
        #     condition_keys_ = None
        #     condition_encodings = None
        #     conditions_combined_ = None
        #     conditions_combined_encodings = None
    # cell pairing
    # if args.cell_pairing_dir:
    #     cell_pairing = read_dataset_files(args.cell_pairing_dir, 'pkl')
    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    if args.split:
        train_indices, val_indices, test_indices = dataset_split(
            tgt_dataset=tgt_dataset,
            # condition_keys=['disease', 'dataset_id'],
            train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
            test_prop=args.test_prop,
            seed=args.seed,
            mode=args.splitting_mode,
        )
    else:
        # return all the indices
        train_indices = list(range(len(tgt_dataset)))
        val_indices = None
        test_indices = list(range(len(tgt_dataset)))
        # # check if the train indices are the same for both adata and dataset
        # if tgt_adata_tmp:
        #     subset_adata = tgt_adata_tmp[train_indices]
        #     subset_dataset = tgt_datasets[
        #         f'tgt_dataset_t{args.n_task_conditions}'
        #     ].select(train_indices)
        #     assert (
        #         subset_adata.obs['cell_pairing_index'].tolist()
        #         == subset_dataset['cell_pairing_index']
        #     )

    print('Data loaded and preprocessed.')
    # Initialize model module
    # ----------------------------------------------------------------------------------


    if args.train_mode == 'masking':
        pretrained_module = CellGenTrainer(
            # tgt_vocab_size=1820,  # 704 for degs, 1820 for tokenised
            tgt_vocab_size=args.tgt_vocab_size,  # max token id + 1 for padding
            d_model=512,
            num_heads=8,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_length=args.max_len + 100,
            dropout=args.cellgen_dropout,
            mlm_probability=args.mlm_prob,
            weight_decay=args.cellgen_wd,
            lr=args.cellgen_lr,
            # lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            mapping_dict_path=args.mapping_dict_path,
            output_dir=args.output_dir,
            encoder_type=args.encoder_type,
            moe_type=args.moe_type,
            alpha=args.alpha,
            apply_attn_mask=True,
            n_task_conditions=args.n_task_conditions,
            num_perturbations=num_perturbations,
        )
        
    elif args.train_mode == 'count':
        decoder_module = CountDecoderTrainer(
            ckpt_masking_path=args.ckpt_masking_path,
            ckpt_count_path=None,
            tgt_vocab_size=args.tgt_vocab_size,
            d_model=512,
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
            tgt_adata=tgt_adata,
            # time_steps=args.time_steps,
            # temperature=args.temperature,
            # iterations=args.iterations,
            # mask_scheduler=args.mask_scheduler,
            output_dir=args.output_dir,
            encoder_type=args.encoder_type,
            seed=args.seed,
            # apply_attn_mask=False,
        )
        
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')
    


    data_module.setup(stage='fit')  # Manually call setup to initialize attributes

    # Now we can access num_perturbations
    num_perturbations = data_module.num_perturbations

    # Initialize data module
    # ----------------------------------------------------------------------------------

    if args.train_mode == 'masking':
        data_module = CellGenDataModule(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            # pairing_metadata=pairing_metadata,
            src_counts=src_counts,  # TODO: do not pass counts in datamodule
            tgt_counts_dict=tgt_counts_dict,
            tgt_adata=tgt_adata,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            split=args.split,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            var_list=args.var_list,
            num_perturbations=num_perturbations,
        )
        
    elif args.train_mode == 'count':
        data_module = CellGenDataModule(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            tgt_adata=tgt_adata,
            src_counts=src_counts,
            tgt_counts_dict=tgt_counts_dict,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=args.shuffle,
            max_len=args.max_len,
            # condition_keys=condition_keys_,
            # condition_encodings=condition_encodings,
            # conditions=conditions,
            # conditions_combined=conditions_combined,
            split=args.split,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            # train_dict=train_dict,
            # val_dict=val_dict,
            # test_dict=test_dict,
            var_list=args.var_list,
            tokenid_to_genename_dict=tokenid_to_genename_dict,
        )
        
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_cellgen')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.

    if args.train_mode == 'masking':
        filename = (
            f'{run_id}_train_{args.train_mode}_lr_{args.cellgen_lr}_'
            f'wd_{args.cellgen_wd}_batch_{args.batch_size}_'
            f'mlmp_{args.mlm_prob}_ntask_{args.n_task_conditions}_s_{args.seed}'
        )
        if args.split:
            monitor_metric = 'val/perplexity'
        else:
            monitor_metric = 'train/masking_loss'
        mode = 'min'
    elif args.train_mode == 'count':
        filename = (
            f'{run_id}_train_{args.train_mode}_lr_{args.count_lr}_wd_{args.count_wd}_'
            f'batch_{args.batch_size}_'
            f'{args.loss_mode}_ntask_{args.n_task_conditions}_s_{args.seed}'
        )
        if args.split:
            monitor_metric = 'val/mse'
            mode = 'min'
        else:
            monitor_metric = 'train/mse'
            mode = 'min'

    checkpoint_path = './T_perturb/Model/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=f'{filename}-' + '{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=10,
        verbose=True,
        # monitor=monitor_metric,
        mode=mode,
    )
    
    ### TEMPORARILY TAKING OUT WANDB ####
    
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
    # advanced_profiler = AdvancedProfiler(
    #     dirpath='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/plt/misc/profiler',
    #     filename='profiler',
    # )
    # device_stats = DeviceStatsMonitor()
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
        gradient_clip_algorithm='norm',
        # precision='16',
    )
    print('Starting training...')
    if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
        # set working directory to root of repository
        os.chdir(
            '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/'
        )
        print('Changed working directory to root of repository')

    if args.train_mode == 'masking':
        # Finally, kick off the training process.
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