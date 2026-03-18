import os
import unittest
from typing import Optional

import pandas as pd
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
import torch
import tempfile
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.Model.trainer import PerturbGenTrainer, CountDecoderTrainer
from perturbgen.src.utils import label_encoder
from perturbgen.tests.utils import dummy_dataset, dummy_cell_gene_matrix




class PerturbGenTestTrainingCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PerturbGenTestTrainingCase, self).__init__(*args, **kwargs)
        self.pred_tps = [1, 2]
        self.n_total_tps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 101  # +1 for padding token
        self.num_genes = self.tgt_vocab_size - 1
        self.batch_size = 4
        self.d_model = 768
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
        base_config = {
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "num_heads": 4,
            "num_layers": 1,
            "d_ff": 8,
            "max_seq_length": self.max_seq_length + 10,
            "dropout": 0.0,
            "pred_tps": self.pred_tps,
            "n_total_tps": self.n_total_tps,
            
            "mapping_dict_path": "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/token_id_to_genename_5000_hvg.pkl",
            "output_dir": "./T_perturb/perturbgen/tests/res",
            "encoder": "scmaskgit",
            "encoder_path": "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt",
            "var_list": ["cell_type"],
            "context_mode": True,
            "mask_scheduler": "pow",
            "seed": 42,
        }
        masking_config = {
            **base_config,
            "end_lr": 1e-3,
            "weight_decay": 0.0,
            "return_embeddings": False,
            "tokenid_to_rowid_path": "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/tokenid_to_rowid_5000_hvg.pkl",
        }
        self.masking_config = masking_config
        countdecoder_config = {
            **base_config,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "loss_mode": "zinb",
            "n_genes": self.num_genes,
            "conditions": conditions_,
            "conditions_combined": conditions_combined_,
        }
        self.countdecoder_config = countdecoder_config

        base_data_config = {
            "src_dataset": src_dataset,
            "tgt_datasets": tgt_datasets,
            "batch_size": self.batch_size,
            "num_workers": 1,
            "pred_tps": self.pred_tps,
            "n_total_tps": self.n_total_tps,
            "max_len": self.max_seq_length,
            "train_indices": np.random.choice(self.num_samples, int(0.8 * self.num_samples), replace=False),
            "use_weighted_sampler": False,
        }
        self.mask_data_config = base_data_config

        base_data_config.update(
            {
                "src_counts": src_counts,
                "tgt_counts_dict": tgt_counts_dict,
                "condition_keys": condition_keys_,
                "condition_encodings": condition_encodings,
                "conditions": conditions,
                "conditions_combined": conditions_combined,
                "use_weighted_sampler": False,
            }
        )
        self.countdecoder_data_config = base_data_config

    def get_transformer(self, tmpdir=None):
        if tmpdir is not None:
            self.masking_config.update({"output_dir": tmpdir})
            
        return PerturbGenTrainer(
            **self.masking_config
        )
    
    def get_countdecoder(self, tmpdir, ckpt_masking_path=None):
        if tmpdir is not None:
            self.countdecoder_config.update({"output_dir": tmpdir})
        self.countdecoder_config.update({"ckpt_masking_path": ckpt_masking_path})
        return CountDecoderTrainer(
            **self.countdecoder_config,
        )
    
    def setup_mask_datamodule(self):
        # Load the data module
        data_module = PerturbGenDataModule(
            **self.mask_data_config
        )
        data_module.setup()

        return data_module
    
    def setup_countdecoder_datamodule(self):
        # Load the data module
        data_module = PerturbGenDataModule(
            **self.countdecoder_data_config
        )
        data_module.setup()

        return data_module

    def test_train_dataloader(self):
        # Access and iterate over the train dataloader
        masking_data_module = self.setup_mask_datamodule()
        train_loader = masking_data_module.train_dataloader()
        self.assertIsNotNone(train_loader, 'Train dataloader should not be None')

        # Test iterating over the dataloader for single batch
        for batch in train_loader:
            self.assertIsNotNone(batch, 'Batch should not be None')
            break

    def test_transformer_forward(self, tmpdir=None):
        # Test forward pass
        masking_data_module = self.setup_mask_datamodule()
        transformer = self.get_transformer(tmpdir)
        batch = next(iter(masking_data_module.train_dataloader()))
        outputs, _ = transformer.forward(batch, generate=False)
        print('batch completed')
        t = list(outputs.keys())[0]
        self.assertEqual(
            outputs[t]['dec_embedding'].shape,
            (
                self.batch_size,
                self.max_seq_length,
                self.d_model,
            ),
        )

    def train_mask_then_count_model(self, tmpdir, mode='all'):
        
        # 1. Train the model using MLM loss for 1 epoch and save a checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=tmpdir,
            filename="mask-checkpoint",
            save_top_k=1,
            monitor=None,   # no metric dependency
            save_last=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,  # fast
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
        )
        transformer = self.get_transformer(tmpdir)
        mask_data_module = self.setup_mask_datamodule()
        trainer.fit(transformer, mask_data_module)
        ckpt_path = checkpoint_callback.last_model_path

        self.assertTrue(
            os.path.exists(ckpt_path),
            "Mask checkpoint should be created",
        )
        if mode == 'masking_only':
            return ckpt_path
        else:
            # 2. Load checkpoint for count prediction
            count_decoder = self.get_countdecoder(tmpdir, ckpt_masking_path=ckpt_path)
            count_data_module = self.setup_countdecoder_datamodule()
            trainer_count = pl.Trainer(
                max_epochs=1,
                limit_train_batches=1,
                logger=False,
                enable_checkpointing=True,
                callbacks=[checkpoint_callback],
            )

            trainer_count.fit(count_decoder, count_data_module)        
            count_ckpt_path = checkpoint_callback.last_model_path
            self.assertTrue(
                os.path.exists(count_ckpt_path),
                "Count decoder checkpoint should be created",
            )
            return count_ckpt_path

    def test_mask_then_count_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = self.train_mask_then_count_model(tmpdir)
            self.assertTrue(os.path.exists(ckpt))
        print("Masking and CountDecoder training workflow test completed successfully.")

    def test_countdecoder_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train the model and get checkpoint
            ckpt_path = self.train_mask_then_count_model(tmpdir, mode='masking_only')

            # Load count decoder with masking checkpoint
            count_decoder = self.get_countdecoder(tmpdir, ckpt_masking_path=ckpt_path)
            count_data_module = self.setup_countdecoder_datamodule()

            batch = next(iter(count_data_module.train_dataloader()))
            outputs, _ = count_decoder.forward(batch)
            t = list(outputs.keys())[0]

            self.assertEqual(
                outputs[t]["count_mean"].shape,
                (
                    self.batch_size,
                    self.num_genes,
                ),
            )
            print("CountDecoder forward pass test completed successfully.")

    # def test_training_loop_with_checkpoint(self):
    #     self.test_transformer_training_loop(save_checkpoint=True)

    # def test_count_forward(self):
    #     transformer = self.get_transformer()
    #     count_data_module = self.setup_countdecoder_datamodule()
    #     # Test that the training loop runs for the expected number of epochs
    #     trainer = pl.Trainer(
    #         max_epochs=1,
    #         enable_checkpointing=False,
    #     )
    #     trainer.fit(transformer, count_data_module)
    #     self.assertEqual(trainer.current_epoch, 1, 'Training should run for 2 epochs')

if __name__ == '__main__':
    unittest.main()
