import os
import unittest
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.Model.trainer import CellGenTrainer


import random
from datasets import load_from_disk


os.chdir('/lustre/scratch126/cellgen/team361/chang/CellGen/')
print('Changed working directory to root of repository')

csv_logger = CSVLogger('T_perturb/tests/norman', name='test_cellgen_training')


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
        cell_idx = np.arange(num_samples)
        dataset = Dataset.from_dict(
            {
                'input_ids': input_ids,
                'length': [len(input_ids)] * num_samples,
                'cell_pairin_index': cell_idx,
            }
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
        self.pred_tps = [1]
        self.n_total_tps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 101  # +1 for padding token
        self.batch_size = 4
        self.d_model = 12

    def setUp(self):
        # Reproducibility
        pl.seed_everything(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        # SUBSETTING ON DATASET TO MAKE TRAINING FASTER SUBSET XYZ

        src_dataset_path = '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_src_random_pairing/control.dataset'
        tgt_dataset_path = '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing/perturbation.dataset'

        tgt_dataset_file = load_from_disk(tgt_dataset_path)
        src_dataset_file = load_from_disk(src_dataset_path)

        random.seed(42)
        subset_size = 5000
        total_samples = len(tgt_dataset_file)

        # Generate random indices
        subset_indices = random.sample(range(total_samples), subset_size)

        # Use the same indices for both datasets
        src_subset_indices = subset_indices
        tgt_subset_indices = subset_indices

        src_dataset = src_dataset_file.select(src_subset_indices)
        tgt_datasets = tgt_dataset_file.select(tgt_subset_indices)


        perturbation_to_index = {}
        current_index=0
        for sample in tgt_datasets:

            perturbation_info = sample.get('perturbation', [])
            if perturbation_info:
                if isinstance(perturbation_info, str):
                    perturbation_str = perturbation_info
                elif isinstance(perturbation_info, list):
                    perturbation_str = '+'.join(sorted(perturbation_info))
            else:
                perturbation_str = 'no_perturbation'  # For controls or unperturbed samples

            if perturbation_str not in perturbation_to_index:
                perturbation_to_index[perturbation_str] = current_index
                current_index += 1

        num_perturbations = current_index




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
            initial_lr=1e-3,
            precision='high',
            pred_tps=self.pred_tps,
            n_total_tps=self.n_total_tps,
            pos_encoding_mode='time_pos_sin',
            encoder='Transformer_encoder',
            mapping_dict_path=('/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl'),
            num_perturbations=num_perturbations
        )
        self.transformer = transformer

        # Load the data module
        self.data_module = CellGenDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            batch_size=self.batch_size,
            num_workers=1,
            pred_tps=self.pred_tps,
            n_total_tps=self.n_total_tps,
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
            dirpath='/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/tests/checkpoints',
            filename='test_masking_checkpoint-{epoch:02d}',
            save_top_k=-1,
            monitor='train/perplexity',
            mode='min',
            every_n_epochs=1,
        )
        if save_checkpoint:
            if not os.path.exists(checkpoint_callback.dirpath):
                os.makedirs(checkpoint_callback.dirpath)
        # Use the PyTorch Lightning Trainer to test the training loop
        trainer = pl.Trainer(
            max_epochs=3,
            logger=csv_logger,
            enable_checkpointing=save_checkpoint,
            callbacks=[checkpoint_callback] if save_checkpoint else [],
        )
        trainer.fit(self.transformer, self.data_module)
        print('finished training')

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