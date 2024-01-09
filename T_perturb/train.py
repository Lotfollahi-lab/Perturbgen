"""Script for training a classifier on  with Pytorch Lightning."""

import argparse
import os


from datetime import datetime
from utils import load_model
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from data_module import GeneseqDataModule
from clipgen_trainer import Clipgenetrainer
import wandb

RANDOM_SEED = 42


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument(
        '--epochs', type=int, default=4, help='number of training epochs'
    )
    parser.add_argument(
        '--model_id', type=str, default='resnet34', help='model id for torch hub'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data', help='path to data directory'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--n_cls', type=int, default=10, help='number of classes')
    parser.add_argument('--n_workers', type=int, default=2, help='number of workers')
    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = get_args()

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)

    # Initialize model module
    # ----------------------------------------------------------------------------------
    gene_encoder = load_model('Pretrained', 0,
                              "/lustre/scratch126/cellgen/team205/ml19/Arian/Geneformer/geneformer-6L-30M/")
    model_module = Clipgenetrainer(
        gene_encoder=gene_encoder,
        weight_decay=args.wd,
    )


    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    data_module = GeneseqDataModule(
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )

    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('clip_geneseq_%Y_%m_%d_%H_%M')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(os.path.join(os.getcwd(),log_path) , exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.
    checkpoint_callback = ModelCheckpoint(
        dirpath="/lustre/scratch126/cellgen/team205/av13/clipgeneseq/model",
        filename='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='train/loss',
        mode='min',
    )
    # Early stopping interrupts training, if there was no improvement in validation
    # loss for a certain training period.
    early_stopping_callback = EarlyStopping(monitor='val/loss', patience=5)

    # The tensorboard logger allows for monitoring the progress of training
    if torch.cuda.device_count() > 1:
        # multi gpu training with group logging
        wandb.init(
            entity='amirh-vahidi',
            project='clip_gene_seq',
            # id=unique_id,  # specify id to log to same run
            group=log_path ,  # all runs are saved in one group for multi gpu training
            dir="/lustre/scratch126/cellgen/team205/av13/clipgeneseq"
        )
    else:
        wandb.init(
            entity='amirh-vahidi',
            project='clip_gene_seq',
            id=run_id ,
            dir="/lustre/scratch126/cellgen/team205/av13/clipgeneseq",
        )

    wandb_logger = WandbLogger(
        log_model="all"
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
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=10),checkpoint_callback],
        max_epochs=args.epochs,
        accelerator=accelerator,
    )

    # Finally, kick of the training process.
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    main()