import os
import unittest

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch

from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.Model.trainer import CellGenTrainer
from T_perturb.src.utils import label_encoder
from T_perturb.tests.test_cellgen_training import dummy_dataset
from T_perturb.tests.test_countdecoder_training import dummy_cell_gene_matrix

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')


class CellGenTestEmbeddingCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CellGenTestEmbeddingCase, self).__init__(*args, **kwargs)
        self.time_step = [1, 2]
        self.total_time_steps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 101  # +1 for padding token
        self.num_genes = self.tgt_vocab_size - 1
        self.batch_size = 4
        self.d_model = 12
        self.num_samples = 100

    def setUp(self):
        pl.seed_everything(42)
        # set conditions and conditions_combined to None if no batch effect
        conditions = None
        condition_keys = None
        conditions_combined = None
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
        tgt_counts_dict = dummy_cell_gene_matrix(
            num_cells=self.num_samples,
            num_genes=self.num_genes,
            total_time_steps=self.total_time_steps,
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

        decoder_module = CellGenTrainer(
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=4,
            num_layers=1,
            d_ff=8,
            max_seq_length=self.max_seq_length + 10,
            lr=1e-3,
            weight_decay=0.0,
            # lr_scheduler_patience=5.0,
            # lr_scheduler_factor=0.8,
            return_embeddings=True,
            dropout=0.0,
            time_steps=self.time_step,
            total_time_steps=2,
            mapping_dict_path='./T_perturb/T_perturb/pp/res/'
            'cytoimmgen/token_id_to_genename_hvg.pkl',
            output_dir='./T_perturb/T_perturb/tests/res',
            mode='Transformer_encoder',
            var_list=None,
        )
        self.decoder_module = decoder_module
        # Load the data module
        self.data_module = CellGenDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            tgt_counts_dict=tgt_counts_dict,
            batch_size=self.batch_size,
            num_workers=1,
            time_steps=[1, 2],
            total_time_steps=self.total_time_steps,
            max_len=self.max_seq_length,
            train_indices=None,
            test_indices=np.random.choice(100, 20, replace=False),
            condition_keys=condition_keys_,
            condition_encodings=condition_encodings,
            conditions=conditions,
            conditions_combined=conditions_combined,
        )
        self.data_module.setup()

    def test_test_dataloader(self):
        # Access and iterate over the test dataloader
        test_loader = self.data_module.test_dataloader()
        self.assertIsNotNone(test_loader, 'Test dataloader should not be None')

        # Test iterating over the dataloader for single batch
        for batch in test_loader:
            self.assertIsNotNone(batch, 'Batch should not be None')
            break

    def test_return_embedding(self):
        # Test generation
        # Use the PyTorch Lightning Trainer to test the training loop
        trainer = pl.Trainer(
            limit_test_batches=1,  # Limit to a single batch for quick testing
            logger=False,
        )
        trainer.test(
            self.decoder_module,
            self.data_module,
            ckpt_path='./T_perturb/T_perturb/tests/'
            'checkpoints/baseline_masking_checkpoint-epoch=00.ckpt',
        )
