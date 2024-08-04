import os
import unittest

# import numpy as np
import pytorch_lightning as pl
import torch

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import CountDecodertrainer
from T_perturb.tests.test_cellgen_training import dummy_src_dataset, dummy_tgt_dataset

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')


# create cell x gene matrix with 100 cells and 100 genes
def dummy_cell_gene_matrix(num_cells: int = 100, num_genes: int = 100):
    return torch.randn(num_cells, num_genes)


class PetraTestTrainingCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PetraTestTrainingCase, self).__init__(*args, **kwargs)
        self.time_step = [1, 2]
        self.total_time_steps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 100
        self.batch_size = 4
        self.d_model = 12

    def setUp(self):
        pl.seed_everything(42)

        decoder_module = CountDecodertrainer(
            ckpt_masking_path='./T_perturb/T_perturb/tests/checkpoints/'
            'test_masking_checkpoint-epoch=00.ckpt',
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=4,
            num_layers=1,
            d_ff=8,
            max_seq_length=self.max_seq_length + 10,
            loss_mode='zinb',
            lr=1e-3,
            weight_decay=0.0,
            # lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            conditions='Cell_culture_batch',
            conditions_combined=None,
            dropout=0.0,
            time_steps=self.time_step,
            total_time_steps=2,
            temperature=1.5,
            iterations=19,
            mask_scheduler='pow',
            output_dir='./T_perturb/T_perturb/plt/res/cytoimmgen',
            mode='Transformer_encoder',
            seed=42,
        )
        self.decoder_module = decoder_module

        # Create dummy data for training
        src_dataset = dummy_src_dataset(
            max_len=self.max_seq_length,
            src_vocab_size=self.tgt_vocab_size,
            num_samples=100,
        )
        tgt_datasets = dummy_tgt_dataset(
            max_len=self.max_seq_length,
            tgt_vocab_size=self.tgt_vocab_size,
            num_samples=100,
        )

        # Load the data module
        self.data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            batch_size=self.batch_size,
            src_counts=dummy_cell_gene_matrix(),
            tgt_counts=dummy_cell_gene_matrix(),
            num_workers=1,
            time_steps=[1, 2],
            total_time_steps=2,
            max_len=self.max_seq_length,
        )
        self.data_module.setup()

    def test_train_dataloader(self):
        # Access and iterate over the train dataloader
        train_loader = self.data_module.train_dataloader()
        self.assertIsNotNone(train_loader, 'Train dataloader should not be None')
