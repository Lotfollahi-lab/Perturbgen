import os
import unittest
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.Model.trainer import CellGenTrainer

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')


def dummy_dataset(
    max_len: int = 50,
    vocab_size: int = 100,
    num_samples: int = 100,
    total_time_steps: Optional[int] = None,
):
    if total_time_steps is None:
        # Generate unique indices for each sample using NumPy
        input_ids_np = np.array(
            [
                np.random.choice(vocab_size, max_len, replace=False)
                for _ in range(num_samples)
            ]
        )
        input_ids = torch.tensor(input_ids_np, dtype=torch.long)
        input_ids[:, -10:] = 0
        dataset = Dataset.from_dict(
            {'input_ids': input_ids, 'length': [len(input_ids)] * num_samples}
        )
        return dataset
    else:
        tgt_dataset_dict = {}
        for t in range(total_time_steps):
            input_ids_np = np.array(
                [
                    np.random.choice(vocab_size, max_len, replace=False)
                    for _ in range(num_samples)
                ]
            )
            input_ids = torch.tensor(input_ids_np, dtype=torch.long)
            input_ids[:, -10:] = 0
            tgt_dataset_dict[f'tgt_dataset_t{t+1}'] = Dataset.from_dict(
                {
                    'input_ids': input_ids,
                    'length': [len(input_ids)] * num_samples,
                    'cell_pairing_index': np.random.choice(
                        100, num_samples, replace=False
                    ),
                }
            )
        return tgt_dataset_dict


class CellGenTestTrainingCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CellGenTestTrainingCase, self).__init__(*args, **kwargs)
        self.time_step = [1, 2]
        self.total_time_steps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 101  # +1 for padding token
        self.batch_size = 4
        self.d_model = 12

    def setUp(self):
        pl.seed_everything(42)

        # Load transformer model and count decoder
        transformer = CellGenTrainer(
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
            total_time_steps=self.total_time_steps,
            mode='Transformer_encoder',
        )
        self.transformer = transformer

        # Create dummy data for training
        src_dataset = dummy_dataset(
            max_len=self.max_seq_length,
            vocab_size=self.tgt_vocab_size,
            num_samples=100,
        )
        tgt_datasets = dummy_dataset(
            max_len=self.max_seq_length,
            vocab_size=self.tgt_vocab_size,
            num_samples=100,
            total_time_steps=self.total_time_steps,
        )

        # Load the data module
        self.data_module = CellGenDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            batch_size=self.batch_size,
            num_workers=1,
            time_steps=[1, 2],
            total_time_steps=self.total_time_steps,
            max_len=self.max_seq_length,
            train_indices=np.random.choice(100, 80, replace=False),
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
        print('batch completed')
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
        print('finished training')

        self.assertEqual(
            trainer.current_epoch, 1, 'Trainer should have completed one epoch'
        )
        if save_checkpoint:
            # Check if the checkpoint was saved
            self.assertTrue(
                checkpoint_callback.best_model_path, 'Checkpoint should be saved'
            )
            self.assertTrue(
                checkpoint_callback.best_model_score, 'Checkpoint should have a score'
            )

    def test_training_loop_with_checkpoint(self):
        self.test_transformer_training_loop(save_checkpoint=True)

    # def test_training_loop_without_checkpoint(self):
    #     self.test_transformer_training_loop(save_checkpoint=False)


if __name__ == '__main__':
    unittest.main()
