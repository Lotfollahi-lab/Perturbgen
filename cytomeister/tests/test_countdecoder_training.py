import os
import unittest
from typing import Optional

import numpy as np
import pandas as pd

# import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from cytomeister.Dataloaders.datamodule import CytoMeisterDataModule
from cytomeister.Model.trainer import CountDecoderTrainer
from cytomeister.src.utils import label_encoder
from cytomeister.tests.test_cellgen_training import dummy_dataset

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/lotfollahi/kl11/t_generative/')

# initialize the logger
csv_logger = CSVLogger(
    'T_perturb/cytomeister/tests/res', name='test_countdecoder_training'
)


# create cell x gene matrix with 100 cells and 100 genes
def dummy_cell_gene_matrix(
    num_cells: int = 100,
    num_genes: int = 100,
    total_time_steps: Optional[int] = None,
):
    lambda_param = 10
    if total_time_steps is None:
        gex_matrix = np.random.poisson(lambda_param, (num_cells, num_genes))
        gex_matrix = gex_matrix.astype(np.float32)
        return np.expand_dims(gex_matrix, axis=1)
    else:
        counts_dict = {}
        for t in range(total_time_steps):
            gex_matrix = np.random.poisson(lambda_param, (num_cells, num_genes))
            gex_matrix = gex_matrix.astype(np.float32)
            counts_dict[f'tgt_h5ad_t{t+1}'] = np.expand_dims(gex_matrix, axis=1)
        return counts_dict


class CytoMeisterTestTrainingCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CytoMeisterTestTrainingCase, self).__init__(*args, **kwargs)
        self.pred_tps = [1, 2]
        self.n_total_tps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 101  # +1 for padding token
        self.num_genes = self.tgt_vocab_size - 1
        self.batch_size = 4
        self.d_model = 12
        self.num_samples = 100

    def setUp(self):
        # Reproducibility
        pl.seed_everything(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # set conditions and conditions_combined to None if no batch effect
        conditions = None
        condition_keys = None
        conditions_combined = None

        # Create dummy data for training
        src_dataset = dummy_dataset(
            max_len=self.max_seq_length,
            vocab_size=self.tgt_vocab_size,
            num_samples=100,
        )
        src_counts = dummy_cell_gene_matrix(
            num_cells=self.num_samples,
            num_genes=self.num_genes,
        )
        tgt_datasets = dummy_dataset(
            max_len=self.max_seq_length,
            vocab_size=self.tgt_vocab_size,
            num_samples=100,
            total_time_steps=self.n_total_tps,
        )
        tgt_counts_dict = dummy_cell_gene_matrix(
            num_cells=self.num_samples,
            num_genes=self.num_genes,
            total_time_steps=self.n_total_tps,
        )

        if condition_keys is None:
            condition_keys = 'tmp_batch'
            # create a mock vector if there are no batch effect
            tmp_series = pd.DataFrame(
                {
                    condition_keys: np.ones(self.num_samples),
                }
            )

        if isinstance(condition_keys, str):
            condition_keys_ = [condition_keys]
        else:
            condition_keys_ = condition_keys

        if conditions is None:
            if condition_keys is not None:
                conditions_ = {}
                for cond in condition_keys_:
                    conditions_[cond] = tmp_series[cond].unique().tolist()
            else:
                conditions_ = {}
        else:
            conditions_ = conditions

        if conditions_combined is None:
            if len(condition_keys_) > 1:
                tmp_series['conditions_combined'] = tmp_series[condition_keys].apply(
                    lambda x: '_'.join(x), axis=1
                )
            else:
                tmp_series['conditions_combined'] = tmp_series[condition_keys]
            conditions_combined_ = tmp_series['conditions_combined'].unique().tolist()
        else:
            conditions_combined_ = conditions_combined

        condition_encodings = {
            cond: {
                k: v for k, v in zip(conditions_[cond], range(len(conditions_[cond])))
            }
            for cond in conditions_.keys()
        }
        conditions_combined_encodings = {
            k: v for k, v in zip(conditions_combined_, range(len(conditions_combined_)))
        }

        tgt_adata_tmp = sc.AnnData(
            X=tgt_counts_dict['tgt_h5ad_t1'].squeeze(), obs=tmp_series
        )

        if (condition_encodings is not None) and (condition_keys_ is not None):
            conditions = [
                label_encoder(
                    tgt_adata_tmp,
                    encoder=condition_encodings[condition_keys_[i]],
                    condition_key=condition_keys_[i],
                )
                for i in range(len(condition_encodings))
            ]
            conditions = torch.tensor(conditions, dtype=torch.long).T
            conditions_combined = label_encoder(
                tgt_adata_tmp,
                encoder=conditions_combined_encodings,
                condition_key='conditions_combined',
            )
            conditions_combined = torch.tensor(conditions_combined, dtype=torch.long)

        decoder_module = CountDecoderTrainer(
            ckpt_masking_path='T_perturb/cytomeister/tests/checkpoints/'
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
            n_genes=self.num_genes,
            # lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            conditions=conditions_,
            conditions_combined=conditions_combined_,
            dropout=0.0,
            pred_tps=self.pred_tps,
            n_total_tps=self.n_total_tps,
            temperature=1.5,
            iterations=19,
            precision='high',
            mask_scheduler='pow',
            output_dir='./T_perturb/cytomeister/plt/res/',
            encoder='Transformer_encoder',
            seed=42,
        )
        self.decoder_module = decoder_module

        # Load the data module
        self.data_module = CytoMeisterDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            batch_size=self.batch_size,
            src_counts=src_counts,
            tgt_counts_dict=tgt_counts_dict,
            num_workers=1,
            pred_tps=self.pred_tps,
            n_total_tps=self.n_total_tps,
            max_len=self.max_seq_length,
            train_indices=np.random.choice(100, 80, replace=False),
            condition_keys=condition_keys_,
            condition_encodings=condition_encodings,
            conditions=conditions,
            conditions_combined=conditions_combined,
        )
        self.data_module.setup()

    def test_train_dataloader(self):
        # Access and iterate over the train dataloader
        train_loader = self.data_module.train_dataloader()
        self.assertIsNotNone(train_loader, 'Train dataloader should not be None')

    def test_countdecoder_forward(self):
        batch = next(iter(self.data_module.train_dataloader()))
        output, _ = self.decoder_module(batch)
        self.assertEqual(
            len(output.keys()),
            self.n_total_tps,
            'Output should contain the same number of keys as time steps',
        )
        self.assertEqual(
            output['count_output_t1']['count_mean'].shape,
            (self.batch_size, self.num_genes),
            'Output shape should be (batch_size, num_genes)',
        )

    def test_countdecoder_training_loop(self, save_checkpoint=False):
        # Setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath='T_perturb/cytomeister/tests/checkpoints',
            filename='test_counts_checkpoint-{epoch:02d}',
            save_top_k=-1,
            every_n_epochs=1,
            monitor='train/mse',
            mode='min',
        )
        if save_checkpoint:
            if not os.path.exists(checkpoint_callback.dirpath):
                os.makedirs(checkpoint_callback.dirpath)
                # Use the PyTorch Lightning Trainer to test the training loop
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,  # Limit to a single batch for quick testing
            logger=csv_logger,
            enable_checkpointing=save_checkpoint,
            callbacks=[checkpoint_callback] if save_checkpoint else [],
        )
        trainer.fit(self.decoder_module, self.data_module)
        if save_checkpoint:
            # Check if the checkpoint was saved
            self.assertTrue(
                checkpoint_callback.best_model_path, 'Checkpoint should be saved'
            )
            self.assertTrue(
                checkpoint_callback.best_model_score, 'Checkpoint should have a score'
            )

    def test_countdecoder_with_checkpoint(self):
        self.test_countdecoder_training_loop(save_checkpoint=True)

    # def test_countdecoder_without_checkpoint(self):
    #     self.test_countdecoder_training_loop(save_checkpoint=False)


if __name__ == '__main__':
    unittest.main()
