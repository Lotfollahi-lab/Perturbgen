import os
import unittest

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Model.trainer import Petratrainer

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')


def dummy_src_dataset(
    max_len: int = 50,
    src_vocab_size: int = 100,
    num_samples: int = 100,
):
    src_input_ids = torch.randint(0, src_vocab_size, (num_samples, max_len))
    src_input_ids[:, -10:] = 0
    src_dataset = Dataset.from_dict(
        {'input_ids': src_input_ids, 'length': [len(src_input_ids)] * num_samples}
    )
    return src_dataset


def dummy_tgt_dataset(
    max_len: int = 10,
    tgt_vocab_size: int = 100,
    num_samples: int = 100,
):
    tgt_input_ids_t1 = torch.randint(0, tgt_vocab_size, (num_samples, max_len))
    tgt_input_ids_t2 = torch.randint(0, tgt_vocab_size, (num_samples, max_len))
    # pad token
    tgt_input_ids_t1[:, -10:] = 0
    tgt_input_ids_t2[:, -10:] = 0
    tgt_dataset_t1 = Dataset.from_dict(
        {
            'input_ids': tgt_input_ids_t1,
            'length': [len(tgt_input_ids_t1)] * num_samples,
            'cell_pairing_index': np.random.choice(100, num_samples, replace=False),
        }
    )
    tgt_dataset_t2 = Dataset.from_dict(
        {
            'input_ids': tgt_input_ids_t2,
            'length': [len(tgt_input_ids_t2)] * num_samples,
            'cell_pairing_index': np.random.choice(100, num_samples, replace=False),
        }
    )
    tgt_dataset = {
        'tgt_dataset_t1': tgt_dataset_t1,
        'tgt_dataset_t2': tgt_dataset_t2,
    }
    return tgt_dataset


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

        # Load transformer model and count decoder
        transformer = Petratrainer(
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=4,
            num_layers=1,
            d_ff=8,
            max_seq_length=self.max_seq_length + 10,  # +10 for special tokens
            dropout=0,
            mlm_probability=0.15,
            weight_decay=0.0,
            lr=1e-3,
            time_steps=self.time_step,
            total_time_steps=2,
            mode='Transformer_encoder',
        )
        self.transformer = transformer

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

        # Test iterating over the dataloader for single batch
        for batch in train_loader:
            self.assertIsNotNone(batch, 'Batch should not be None')
            break

    def test_transformer_forward(self):
        # Test forward pass
        batch = next(iter(self.data_module.train_dataloader()))
        output = self.transformer(batch)
        print(output['dec_embedding'].shape)
        self.assertEqual(
            output['dec_embedding'].shape,
            (
                self.batch_size,
                self.max_seq_length + 1,  # +1 for cls token
                self.d_model,
            ),
        )

    def test_transformer_training_loop(self, save_checkpoint=False):
        # Setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath='T_perturb/T_perturb/tests/checkpoints',
            filename='test_masking_checkpoint-{epoch:02d}',
            save_top_k=1,
            monitor='train/perplexity',
            mode='min',
        )
        if save_checkpoint:
            if not os.path.exists(checkpoint_callback.dirpath):
                os.makedirs(checkpoint_callback.dirpath)
        # Use the PyTorch Lightning Trainer to test the training loop
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,  # Limit to a single batch for quick testing
            logger=False,
            enable_checkpointing=save_checkpoint,
            callbacks=[checkpoint_callback] if save_checkpoint else [],
        )
        trainer.fit(self.transformer, self.data_module)

        self.assertEqual(
            trainer.current_epoch, 1, 'Trainer should have completed one epoch'
        )
        print(checkpoint_callback)
        if save_checkpoint:
            # Check if the checkpoint was saved
            self.assertTrue(
                checkpoint_callback.best_model_path, 'Checkpoint should be saved'
            )
            self.assertTrue(
                checkpoint_callback.best_model_score, 'Checkpoint should have a score'
            )

    # def test_training_loop_with_checkpoint(self):
    #     # Test training loop with checkpoint
    #     self.test_transformer_training_loop(save_checkpoint=True)

    def test_training_loop_without_checkpoint(self):
        # Test training loop without checkpoint
        self.test_transformer_training_loop(save_checkpoint=False)


if __name__ == '__main__':
    unittest.main()
