"""Script for training a classifier on  with Pytorch Lightning."""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from T_perturb.Dataloaders.datamodule import GeneformerDataModule
from T_perturb.Model.trainer import TTransformertrainer

RANDOM_SEED = 100


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dataset_folder',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/'
        'cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset_folder',
        type=str,
        default='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/'
        'cytoimmgen_tokenised_degs_stratified_pairing_16h.dataset',
        help='path to tokenised activated data',
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument(
        '--epochs', type=int, default=5, help='number of training epochs'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--mlm_probability', type=float, default=0.5, help='mlm probability'
    )
    parser.add_argument('--max_len', type=int, default=246, help='max sequence length')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    # parser.add_argument('--n_cls', type=int, default=10, help='number of classes')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers')
    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = get_args()

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)

    # Initialize model module
    # ----------------------------------------------------------------------------------
    model_module = TTransformertrainer(
        tgt_vocab_size=704,
        d_model=256,
        num_heads=8,
        num_layers=1,
        d_ff=32,
        max_seq_length=2000,
        dropout=0.0,
        mlm_probability=args.mlm_probability,
        weight_decay=args.wd,
        lr=args.lr,
        lr_scheduler_patience=1.0,
        lr_scheduler_factor=0.8,
        return_cls_embedding=False,
        generate=True,
    )
    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    # change dataloader and input
    data_module = GeneformerDataModule(
        src_dataset_folder=args.src_dataset_folder,
        tgt_dataset_folder=args.tgt_dataset_folder,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=args.shuffle,
        max_len=args.max_len,
    )

    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime(
        f'%Y%m%d_%H%M_ttransformer_{args.batch_size}_{args.lr}_{args.wd}'
    )
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.
    checkpoint_callback = ModelCheckpoint(
        dirpath='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/Model/checkpoints',
        filename='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='train/loss',
        mode='min',
    )

    # The tensorboard logger allows for monitoring the progress of training
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb.init(
            entity='k-ly',
            project='ttransformer',
            # id=unique_id,  # specify id to log to same run
            group=log_path,  # all runs are saved in one group for multi gpu training
            dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb',
        )
    else:
        wandb.init(
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
        callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
        max_epochs=args.epochs,
        accelerator=accelerator,
    )
    # Finally, kick of the training process.
    trainer.test(
        model_module,
        data_module,
        ckpt_path='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/Model/checkpoints/'
        '20240212_1657_ttransformer_lr_0.001_wd_0.001_'
        'batchsize_64_mlmprob_0.4_16h_pairing_stratified.ckpt',
    )


if __name__ == '__main__':
    main()
