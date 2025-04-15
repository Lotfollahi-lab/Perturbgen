"""Script for training a classifier on with Pytorch Lightning."""

import argparse
import os
import uuid
from datetime import datetime

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import concatenate_datasets, load_from_disk

# from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy  # ,DeepSpeedStrategy

from T_perturb.Dataloaders.datamodule import CytoMeisterDataModule
from T_perturb.Model.trainer import CountDecoderTrainer, CytoMeisterTrainer
from T_perturb.src.utils import (
    condition_for_count_loss,
    randomised_split,
    read_dataset_files,
    str2bool,
    stratified_split,
)

# from pytorch_lightning.utilities.deepspeed import (
#     convert_zero_checkpoint_to_fp32_state_dict,
# )


if os.getcwd().split('/')[-1] != 't_generative':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/kl11/t_generative/')
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
        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
    )
    parser.add_argument(
        '--split_obs',
        type=str,
        nargs='+',
        default=['cell_type_cellgen_harm'],
        # default=['celltype_v2'],
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
    parser.add_argument('--num_node', type=int, default=1)
    parser.add_argument('--use_positional_encoding', type=str2bool, default=False)
    parser.add_argument('--layer_norm', type=str2bool, default=False)
    parser.add_argument('--add_cell_time', type=str2bool, default=False)

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
        '--n_workers', type=int, default=8, help='number of workers'
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
    parser.add_argument(
        '--mask_scheduler',
        type=str,
        default='cosine',
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
        # type=list,
        nargs='+',
        type=str,
        # default=['Time_point'],
        default=['cell_type_cellgen_harm', 'donor_cellgen_harm', 'time_after_LPS'],
        # default=['celltype_v2', 'sex', 'phase', 'tissue', 'diff_state'],
        help='List of variables to keep in the dataset',
    )
    parser.add_argument(
        '--cond_list',
        nargs='+',
        type=str,
        help='List of variables to form condition tokens',
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
            'scmaskgit',
        ],
        help='mode of encoder',
    )
    parser.add_argument(
        '--encoder_path',
        default=None,
        type=str,
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
    parser.add_argument(
        '--d_condc',
        type=int,
        default=1,
        help='One Hot dimension',
    )
    parser.add_argument(
        '--d_condt',
        type=int,
        default=768,
        help='One Hot dimension',
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='embedding dimension',
    )
    parser.add_argument(
        '--sampling_keys',
        nargs='+',
        type=str,
        help='List of variables to form condition tokens',
    )
    parser.add_argument(
        '--use_weighted_sampler',
        type=str2bool,
        default=True,
        help='use weighted sampler',
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
    print('Loading and preprocessing data...')
    tgt_datasets = read_dataset_files(args.tgt_dataset_folder, 'dataset')
    tgt_adatas = read_dataset_files(args.tgt_adata_folder, 'h5ad')
    src_dataset = load_from_disk(args.src_dataset)
    src_adata = sc.read_h5ad(args.src_adata)

    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    tgt_adata_tmp = tgt_adatas[f'tgt_h5ad_t{args.pred_tps[0]}']
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
        # train_indices = list(range(100))
        val_indices = None
        test_indices = list(
            range(len(tgt_datasets[f'tgt_dataset_t{args.pred_tps[0]}']))
        )
    if args.loss_mode == 'mse':
        # log normalize data only for mse loss
        sc.pp.normalize_total(src_adata, target_sum=1e4)
        sc.pp.log1p(src_adata)
        for _, tgt_adata in tgt_adatas.items():
            sc.pp.normalize_total(tgt_adata, target_sum=1e4)
            sc.pp.log1p(tgt_adata)

    # ZINB count loss preprocessing
    # ----------------------------------------------------------------------------------
    if args.train_mode == 'count':
        (
            conditions,
            condition_encodings,
            conditions_combined,
            conditions_,
            condition_keys_,
            conditions_combined_,
        ) = condition_for_count_loss(
            args.condition_keys,
            args.conditions,
            args.conditions_combined,
            tgt_adata_tmp,
        )
    # count number of unique timepoints
    n_total_tps = len(tgt_adatas)

    # create dictionnary of metadata for classifier-free guidance
    token_no = args.tgt_vocab_size

    # create full dataset to extract metadata for conditioning
    dataset_list = [src_dataset]
    for dataset in tgt_datasets.values():
        dataset_list.append(dataset)
    full_dataset = concatenate_datasets(dataset_list)
    if args.cond_list is not None:
        condition_dict = {}
        for condition in args.cond_list:
            condition_dict[condition] = {
                cell_type: i + token_no
                for i, cell_type in enumerate(full_dataset.unique(condition))
            }
            token_no += len(condition_dict[condition])
    else:
        condition_dict = None

    # Initialize model module
    # ----------------------------------------------------------------------------------
    trainer_kwargs = {
        'tgt_vocab_size': token_no + 50,  # add 50 for extra tokens
        'd_model': args.d_model,
        'num_heads': 8,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_seq_length': args.max_len + 100,
        'mask_scheduler': args.mask_scheduler,
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'n_total_tps': n_total_tps,
        'encoder': args.encoder,
        'output_dir': args.output_dir,
        'pos_encoding_mode': args.pos_encoding_mode,
        'encoder_path': args.encoder_path,
        'condition_dict': condition_dict,
        'temperature': args.temperature,
        'iterations': args.iterations,
        'mapping_dict_path': args.mapping_dict_path,
    }
    if args.train_mode == 'masking':
        trainer_kwargs['dropout'] = args.cellgen_dropout
        trainer_kwargs['mlm_probability'] = args.mlm_prob
        trainer_kwargs['end_lr'] = args.cellgen_lr
        trainer_kwargs['weight_decay'] = args.cellgen_wd
        trainer_kwargs['context_mode'] = args.context_mode
        pretrained_module = CytoMeisterTrainer(**trainer_kwargs)
    elif args.train_mode == 'count':
        trainer_kwargs['ckpt_masking_path'] = args.ckpt_masking_path
        trainer_kwargs['ckpt_count_path'] = None
        trainer_kwargs['loss_mode'] = args.loss_mode
        trainer_kwargs['d_condc'] = args.d_condc
        trainer_kwargs['d_condt'] = args.d_condt
        trainer_kwargs['layer_norm'] = args.layer_norm
        trainer_kwargs['use_positional_encoding'] = args.use_positional_encoding
        trainer_kwargs['add_cell_time'] = args.add_cell_time
        trainer_kwargs['lr'] = args.count_lr
        trainer_kwargs['weight_decay'] = args.count_wd
        trainer_kwargs['conditions'] = conditions_
        trainer_kwargs['conditions_combined'] = conditions_combined_
        trainer_kwargs['tgt_adata'] = tgt_adatas
        trainer_kwargs['temperature'] = args.temperature
        trainer_kwargs['iterations'] = args.iterations
        trainer_kwargs['n_genes'] = src_adata.shape[1]
        trainer_kwargs['dropout'] = args.count_dropout
        # trainer_kwargs['use_positional_encoding'] = args.use_positional_encoding
        trainer_kwargs['seed'] = args.seed
        decoder_module = CountDecoderTrainer(**trainer_kwargs)
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
    # determine global batch size to account for multiple GPUs
    gpu_number = max(torch.cuda.device_count(), 1)
    per_gpu_batch_size = args.batch_size // gpu_number
    data_module_kwargs = {
        'src_dataset': src_dataset,
        'tgt_datasets': tgt_datasets,
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
        'n_total_tps': n_total_tps,
        'var_list': args.var_list,
        'use_weighted_sampler': args.use_weighted_sampler,
        'sampling_keys': args.sampling_keys,
        'seed': args.seed,
    }
    if args.train_mode == 'masking':
        # TODO: Do not pass src into DataModule
        data_module = CytoMeisterDataModule(**data_module_kwargs)

    elif args.train_mode == 'count':
        data_module_kwargs['src_counts'] = src_counts
        data_module_kwargs['tgt_counts_dict'] = tgt_counts_dict
        data_module_kwargs['condition_keys'] = condition_keys_
        data_module_kwargs['condition_encodings'] = condition_encodings
        data_module_kwargs['conditions'] = conditions
        data_module_kwargs['conditions_combined'] = conditions_combined
        data_module = CytoMeisterDataModule(**data_module_kwargs)
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
        if val_indices is not None:
            monitor_metric = 'val/perplexity'
        else:
            monitor_metric = 'train/perplexity'
        mode = 'min'
    elif args.train_mode == 'count':
        filename = (
            f'{run_id}_train_{args.train_mode}_lr_{args.count_lr}_wd_{args.count_wd}_'
            f'batch_{args.batch_size}_'
            f'{args.loss_mode}_tp_{time_steps_str}_s_'
            f'{args.seed}_pos_{args.pos_encoding_mode}_m_{args.mask_scheduler}'
        )
        if val_indices:
            monitor_metric = 'val/mse'
            mode = 'min'
        else:
            monitor_metric = 'train/mse'
            mode = 'min'

    checkpoint_path = os.path.join(args.output_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=f'{filename}-' + '{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=5,
        verbose=True,
        monitor=monitor_metric,
        mode=mode,
    )
    # The tensorboard logger allows for monitoring the progress of training
    # Configure WandbLogger with unique name for each run
    run_name = (
        f'{run_id}_{str(uuid.uuid4())[:6]}' if torch.cuda.device_count() > 1 else run_id
    )
    wandb_logger = WandbLogger(
        project='ttransformer', name=run_name, save_dir=args.log_dir, log_model=False
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
    deepspeed_strategy = DeepSpeedStrategy(
        stage=2,
        # offload_optimizer=True,
        # offload_parameters=True,
    )

    # if torch.cuda.is_available():
    #     cuda_device_name = torch.cuda.get_device_name()
    # if ('A100' in cuda_device_name) or ('NVIDIA H100 80GB HBM' in cuda_device_name):
    #     print(f'Using {cuda_device_name} for training')
    #     precision = 'bf16-mixed'
    # else:
    #     precision = '16-mixed'
    # # After each epoch, convert DeepSpeed checkpoint to FP32 and save
    # class DeepSpeedCheckpointConverter(Callback):
    #     def __init__(self, checkpoint_path, filename, save_interval=5):
    #         super().__init__()
    #         self.checkpoint_path = checkpoint_path
    #         print(f'Checkpoint path: {checkpoint_path}')
    #         self.filename = filename
    #         print(f'Filename: {filename}')
    #         self.save_interval = save_interval

    #     def on_train_epoch_end(self, trainer, pl_module):
    #         # Only save at specified intervals
    #         if trainer.current_epoch % self.save_interval == 0:
    #             # Try accessing DeepSpeed through pl_module
    #             deepspeed_engine = getattr(pl_module, 'deepspeed', None)

    #             if isinstance(trainer.strategy, DeepSpeedStrategy):
    #                 # Define DeepSpeed checkpoint path
    #                 deepspeed_epoch_checkpoint_path = os.path.join(
    #                     self.checkpoint_path, f'epoch_{trainer.current_epoch:02d}'
    #                 )

    #                 # Save the checkpoint using DeepSpeed’s save_checkpoint method
    #                 deepspeed_engine.save_checkpoint(deepspeed_epoch_checkpoint_path)
    #                 print(
    #                     f'DeepSpeed checkpoint saved at'
    #                     f'{deepspeed_epoch_checkpoint_path}'
    #                 )

    #                 # Convert the DeepSpeed checkpoint to FP32 format
    #                 fp32_state_dict = os.path.join(
    #                     self.checkpoint_path,
    #                     f'{self.filename}-epoch={trainer.current_epoch:02d}-fp32.pth',
    #                 )
    #                 convert_zero_checkpoint_to_fp32_state_dict(
    #                     zero_checkpoint_path=deepspeed_epoch_checkpoint_path,
    #                     output_path=fp32_state_dict,
    #                 )
    #                 print(f'FP32 checkpoint saved at {fp32_state_dict}')
    #             else:
    #                 print('DeepSpeed not initialized in pl_module')

    # deepspeed_convert_ckpt = DeepSpeedCheckpointConverter(
    #     checkpoint_path=checkpoint_path, filename=filename
    # )
    # If the device is an A100, set the precision for matrix multiplication
    # ddp_strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            early_stop_callback,
            checkpoint_callback,
        ],
        # accumulate_grad_batches=10,
        max_epochs=args.epochs,
        accelerator=accelerator,
        # precision=precision,
        # gradient_clip_val=1.0,
        devices=-1 if torch.cuda.is_available() else 0,
        num_nodes=args.num_node,
        strategy=deepspeed_strategy if torch.cuda.device_count() > 1 else 'auto',
    )

    if args.train_mode == 'masking':
        # Finally, kick of the training process.
        if args.ckpt_masking_path is not None:
            # check if masking_path ends with .bin
            if args.ckpt_masking_path.endswith('.bin'):
                # load the model from the bin file
                state_dict = torch.load(args.ckpt_masking_path, map_location='cpu')
                pretrained_module.load_state_dict(state_dict, strict=False)
                trainer.fit(
                    pretrained_module,
                    data_module,
                )
            else:
                trainer.fit(
                    pretrained_module,
                    data_module,
                    ckpt_path=args.ckpt_masking_path,
                )
        else:
            trainer.fit(pretrained_module, data_module)
    elif args.train_mode == 'count':
        trainer.fit(decoder_module, data_module)
    else:
        raise ValueError('train_mode not recognised, needs to be masking or count')

    # # #collate deepzero checkpoint
    # if torch.cuda.device_count() > 1:
    #     checkpoint_path = os.path.join(
    #         checkpoint_path,
    #         f'{filename}-epoch={trainer.current_epoch}.ckpt'
    #     )
    #     print(f'Saving checkpoint to {checkpoint_path}')
    # # check if checkpoint path exists
    # if os.path.exists(checkpoint_path):

    #     convert_zero_checkpoint_to_fp32_state_dict(
    #         zero_checkpoint_path=checkpoint_path,
    #         output_path=checkpoint_path,
    #         tag='fp32'
    #     )


if __name__ == '__main__':
    main()
