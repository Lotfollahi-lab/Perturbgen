import os
import unittest

import numpy as np
import pytorch_lightning as pl

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import Petratrainer
from T_perturb.tests.test_cellgen_training import dummy_src_dataset, dummy_tgt_dataset

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')


class PetraTestGenerationCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PetraTestGenerationCase, self).__init__(*args, **kwargs)
        self.src_dataset = dummy_src_dataset()
        self.tgt_dataset = dummy_tgt_dataset()

        self.time_step = [1, 2]
        self.total_time_steps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 100
        self.batch_size = 4
        self.d_model = 12

    def setUp(self):
        pl.seed_everything(42)
        model = Petratrainer(
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=4,
            num_layers=1,
            d_ff=14,
            max_seq_length=self.max_seq_length + 10,  # +10 for special tokens
            dropout=0,
            mlm_probability=0.15,
            weight_decay=0.0,
            lr=1e-3,
            time_steps=self.time_step,
            total_time_steps=2,
            mode='Transformer_encoder',
        )
        self.model = model

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
        self.data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            batch_size=self.batch_size,
            num_workers=1,
            time_steps=[1, 2],
            total_time_steps=2,
            train_indices=None,
            test_indices=np.random.choice(100, 20, replace=False),
            max_len=self.max_seq_length,
        )
        self.data_module.setup()

    def test_generation(self):
        # Test generation
        batch = next(iter(self.data_module.test_dataloader()))
        output = self.model.generate(batch)
        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                self.max_seq_length + 1,  # +1 for cls token
            ),
        )


if __name__ == '__main__':
    unittest.main()
