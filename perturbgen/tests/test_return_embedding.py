import os
import unittest

import numpy as np
import pandas as pd
import tempfile
import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import concatenate_datasets

from pytorch_lightning.callbacks import ModelCheckpoint
from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.Model.trainer import PerturbGenTrainer
from perturbgen.src.utils import label_encoder
from perturbgen.tests.test_cellgen_training import dummy_dataset
from perturbgen.tests.test_countdecoder_training import dummy_cell_gene_matrix


class PerturbGenTestEmbeddingCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PerturbGenTestEmbeddingCase, self).__init__(*args, **kwargs)
        self.pred_tps = [1, 2]
        self.context_tps = [1, 2]
        self.n_total_tps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 101  # +1 for padding token
        self.num_genes = self.tgt_vocab_size - 1
        self.batch_size = 4
        self.d_model = 768
        self.num_samples = 100
        self.gene_embs_condition = 'cell_type'

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
        
        if self.gene_embs_condition is not None:
            pred_dataset = {
                key: tgt_datasets[key]
                for key in tgt_datasets
                if key in {f'tgt_dataset_t{tp}' for tp in self.pred_tps}
            }

            all_pred_dataset = concatenate_datasets(list(pred_dataset.values()))

            gene_embs_list = all_pred_dataset.unique(self.gene_embs_condition)

            print(
                f"Return gene embs for {gene_embs_list} "
                f"in {self.gene_embs_condition}."
            )
        else:
            gene_embs_list = None
        self.gene_embs_list = gene_embs_list
            

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
            model_config = {
                "tgt_vocab_size": self.tgt_vocab_size,
                "d_model": self.d_model,
                "num_heads": 4,
                "num_layers": 1,
                "d_ff": 8,
                "max_seq_length": self.max_seq_length + 10,
                "end_lr": 1e-3,
                "weight_decay": 0.0,
                "return_embeddings": False,
                "dropout": 0.0,
                "pred_tps": self.pred_tps,
                "n_total_tps": 3,
                "tokenid_to_rowid_path": "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/tokenid_to_rowid_5000_hvg.pkl",
                "mapping_dict_path": "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/token_id_to_genename_5000_hvg.pkl",
                "output_dir": "./T_perturb/perturbgen/tests/res",
                "encoder": "scmaskgit",
                "encoder_path": "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt",
                "var_list": ["cell_type"],
                "context_mode": True,
                "mask_scheduler": "pow",
            }
        self.model_config = model_config

        # Load the data module
        self.data_module = PerturbGenDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            tgt_counts_dict=tgt_counts_dict,
            batch_size=self.batch_size,
            num_workers=1,
            pred_tps=self.pred_tps,
            n_total_tps=self.n_total_tps+1,
            max_len=self.max_seq_length,
            train_indices=None,
            test_indices=np.arange(20),
            condition_keys=condition_keys_,
            condition_encodings=condition_encodings,
            conditions=conditions,
            conditions_combined=conditions_combined,
            var_list=self.model_config["var_list"],
            use_weighted_sampler=False,
            context_tps=self.context_tps,
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
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_callback = ModelCheckpoint(
                dirpath=tmpdir,
                filename="test-checkpoint",
                save_top_k=1,
                monitor=None,  # no validation metric needed
                save_last=True,
            )
            train_module = PerturbGenTrainer(
                **self.model_config
            )
            self.train_module = train_module
        
        


            trainer = pl.Trainer(
                max_epochs=1,
                limit_train_batches=1,
                limit_test_batches=1,
                logger=False,
                enable_checkpointing=True,
                callbacks=[checkpoint_callback],
            )

            # --- TRAIN (just enough to produce a checkpoint) ---
            trainer.fit(self.train_module, self.data_module)

            # Get the saved checkpoint path
            ckpt_path = checkpoint_callback.last_model_path
            self.assertTrue(os.path.exists(ckpt_path), "Checkpoint was not created")
            self.model_config["generate"] = False
            self.model_config["return_gene_embs"] = True
            
            self.model_config["gene_embs_list"] = self.gene_embs_list
            self.model_config["gene_embs_condition"] = self.gene_embs_condition
            self.model_config["context_tps"] = self.context_tps
            model = PerturbGenTrainer.load_from_checkpoint(
                ckpt_path,
                **self.model_config  # includes gene_embs stuff
            )

            # --- TEST using the generated checkpoint ---
            trainer.test(
                model,
                self.data_module,
            )
