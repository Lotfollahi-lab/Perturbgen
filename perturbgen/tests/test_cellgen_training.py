import os
import unittest
import tempfile
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint

from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.Model.trainer import PerturbGenTrainer, CountDecoderTrainer
from perturbgen.src.utils import label_encoder
from perturbgen.tests.utils import dummy_dataset, dummy_cell_gene_matrix


# =========================
# REQUIRED PATHS (LEAN + EXPLICIT)
# =========================

ENCODER_PATH = "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
MAPPING_PATH = "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/token_id_to_genename_5000_hvg.pkl"
TOKENID_PATH = "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/tokenid_to_rowid_5000_hvg.pkl"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "res")

def resolve_path(path):
    return os.path.abspath(os.path.expanduser(path))

paths = [ENCODER_PATH, MAPPING_PATH, TOKENID_PATH, OUTPUT_DIR]

for path in paths:
    full_path = resolve_path(path)
    print(f"Checking: {path} -> {full_path}")
    assert os.path.exists(full_path), f"Missing required file: {full_path}"

# print working dir and files for debugging
print(f"Working directory: {os.getcwd()}")

# =========================
# Shared Test Utilities
# =========================

class TestConfigMixin:

    @classmethod
    def init_shared_state(cls):
        pl.seed_everything(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        cls.pred_tps = [1, 2]
        cls.n_total_tps = 2
        cls.max_seq_length = 50
        cls.tgt_vocab_size = 101
        cls.num_genes = cls.tgt_vocab_size - 1
        cls.batch_size = 4
        cls.d_model = 768
        cls.num_samples = 100

        cls._build_dummy_data()
        cls._build_configs()

    @classmethod
    def _build_dummy_data(cls):
        cls.src_dataset = dummy_dataset(
            max_len=cls.max_seq_length,
            vocab_size=cls.tgt_vocab_size,
            num_samples=cls.num_samples,
        )

        cls.src_counts = dummy_cell_gene_matrix(
            num_cells=cls.num_samples,
            num_genes=cls.num_genes,
        )

        cls.tgt_datasets = dummy_dataset(
            max_len=cls.max_seq_length,
            vocab_size=cls.tgt_vocab_size,
            num_samples=cls.num_samples,
            total_time_steps=cls.n_total_tps,
        )

        cls.tgt_counts_dict = dummy_cell_gene_matrix(
            num_cells=cls.num_samples,
            num_genes=cls.num_genes,
            total_time_steps=cls.n_total_tps,
        )

        tmp_series = pd.DataFrame({"tmp_batch": np.ones(cls.num_samples)})

        cls.condition_keys = ["tmp_batch"]
        cls.conditions_ = {"tmp_batch": [1.0]}
        cls.conditions_combined_ = [1.0]

        condition_encodings = {"tmp_batch": {1.0: 0}}
        conditions_combined_encodings = {1.0: 0}

        tgt_adata_tmp = sc.AnnData(
            X=cls.tgt_counts_dict['tgt_h5ad_t1'].squeeze(), obs=tmp_series
        )

        cls.conditions = torch.tensor(
            [label_encoder(tgt_adata_tmp, condition_encodings["tmp_batch"], "tmp_batch")],
            dtype=torch.long,
        ).T

        cls.conditions_combined = torch.tensor(
            label_encoder(tgt_adata_tmp, conditions_combined_encodings, "tmp_batch"),
            dtype=torch.long,
        )

    @classmethod
    def _build_configs(cls):
        base_config = {
            "tgt_vocab_size": cls.tgt_vocab_size,
            "d_model": cls.d_model,
            "num_heads": 4,
            "num_layers": 1,
            "d_ff": 8,
            "max_seq_length": cls.max_seq_length + 10,
            "dropout": 0.0,
            "pred_tps": cls.pred_tps,
            "n_total_tps": cls.n_total_tps,
            "encoder": "scmaskgit",
            "encoder_path": ENCODER_PATH,
            "mapping_dict_path": MAPPING_PATH,
            "context_mode": True,
            "mask_scheduler": "pow",
            "seed": 42,
        }

        cls.masking_config = {
            **base_config,
            "end_lr": 1e-3,
            "weight_decay": 0.0,
            "return_embeddings": False,
            "tokenid_to_rowid_path": TOKENID_PATH,
        }

        cls.countdecoder_config = {
            **base_config,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "loss_mode": "zinb",
            "n_genes": cls.num_genes,
            "conditions": cls.conditions_,
            "conditions_combined": cls.conditions_combined_,
        }

    @classmethod
    def build_mask_datamodule(cls):
        dm = PerturbGenDataModule(
            src_dataset=cls.src_dataset,
            tgt_datasets=cls.tgt_datasets,
            batch_size=cls.batch_size,
            num_workers=1,
            pred_tps=cls.pred_tps,
            n_total_tps=cls.n_total_tps,
            max_len=cls.max_seq_length,
            train_indices=np.arange(cls.num_samples),
            use_weighted_sampler=False,
        )
        dm.setup()
        return dm

    @classmethod
    def build_count_datamodule(cls):
        dm = PerturbGenDataModule(
            src_dataset=cls.src_dataset,
            tgt_datasets=cls.tgt_datasets,
            src_counts=cls.src_counts,
            tgt_counts_dict=cls.tgt_counts_dict,
            batch_size=cls.batch_size,
            num_workers=1,
            pred_tps=cls.pred_tps,
            n_total_tps=cls.n_total_tps,
            max_len=cls.max_seq_length,
            condition_keys=cls.condition_keys,
            condition_encodings={"tmp_batch": {1.0: 0}},
            conditions=cls.conditions,
            conditions_combined=cls.conditions_combined,
            use_weighted_sampler=False,
        )
        dm.setup()
        return dm


# =========================
# Checkpoint Fixture
# =========================

class MaskCheckpointFixture(TestConfigMixin):
    _shared_ckpt_path = None
    _shared_tmpdir = None

    @classmethod
    def setUpClass(cls):
        cls.init_shared_state()
        if MaskCheckpointFixture._shared_ckpt_path and \
           os.path.exists(MaskCheckpointFixture._shared_ckpt_path):
            cls.ckpt_path = MaskCheckpointFixture._shared_ckpt_path
            cls.tmpdir = MaskCheckpointFixture._shared_tmpdir
            return

        cls.tmpdir_obj = tempfile.TemporaryDirectory()
        cls.tmpdir = cls.tmpdir_obj.name

        checkpoint_callback = ModelCheckpoint(
            dirpath=cls.tmpdir,
            filename="mask",
            save_last=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        model = PerturbGenTrainer(
            **cls.masking_config,
            output_dir=cls.tmpdir,
        )

        dm = cls.build_mask_datamodule()
        trainer.fit(model, dm)

        cls.ckpt_path = checkpoint_callback.last_model_path
        
        cls.output_dir = os.path.join(OUTPUT_DIR, "res")
        os.makedirs(cls.output_dir, exist_ok=True)

        shutil.copy(cls.ckpt_path, os.path.join(OUTPUT_DIR, "new_mask.ckpt"))
        MaskCheckpointFixture._shared_ckpt_path = cls.ckpt_path
        MaskCheckpointFixture._shared_tmpdir = cls.tmpdir


# =========================
# Tests
# =========================

class TestMasking(TestConfigMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.init_shared_state()

    def test_forward_pass(self):
        dm = self.build_mask_datamodule()
        model = PerturbGenTrainer(**self.masking_config)

        batch = next(iter(dm.train_dataloader()))
        outputs, _ = model.forward(batch, generate=False)

        t = list(outputs.keys())[0]
        self.assertEqual(
            outputs[t]['dec_embedding'].shape,
            (self.batch_size, self.max_seq_length, self.d_model),
        )

class TestEmbedding(MaskCheckpointFixture, unittest.TestCase):

    def test_return_embedding(self):
        embedding_config = self.masking_config.copy()
        embedding_config.update({
            "return_embeddings": True,
            "output_dir": self.tmpdir,
        })
        model = PerturbGenTrainer.load_from_checkpoint(
            self.ckpt_path,
            **embedding_config,
        )

        dm = self.build_mask_datamodule()
        batch = next(iter(dm.train_dataloader()))
        outputs, _ = model.forward(batch, generate=False)

        t = list(outputs.keys())[0]
        self.assertIn('dec_embedding', outputs[t])
        self.assertEqual(
            outputs[t]['dec_embedding'].shape,
            (self.batch_size, self.max_seq_length, self.d_model),
        )
class TestCountDecoder(MaskCheckpointFixture, unittest.TestCase):
    _shared_count_ckpt = None
    @classmethod
    def _build_count_model(cls):
        return CountDecoderTrainer(
            **cls.countdecoder_config,
            ckpt_masking_path=cls.ckpt_path,
            output_dir=cls.tmpdir,
        )
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # builds masking checkpoint
        if TestCountDecoder._shared_count_ckpt and \
            os.path.exists(TestCountDecoder._shared_count_ckpt):
            cls.count_ckpt_path = TestCountDecoder._shared_count_ckpt
            return

        # train count decoder ONCE
        checkpoint_callback = ModelCheckpoint(
            dirpath=cls.tmpdir,
            filename="count",
            save_last=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        model = cls._build_count_model()

        dm = cls.build_count_datamodule()
        trainer.fit(model, dm)
        cls.count_ckpt_path = checkpoint_callback.last_model_path
        shutil.copy(cls.count_ckpt_path, os.path.join(cls.output_dir, "new_count.ckpt"))
        TestCountDecoder._shared_count_ckpt = cls.count_ckpt_path

    def test_forward(self):
        model = CountDecoderTrainer(
            **self.countdecoder_config,
            ckpt_masking_path=self.ckpt_path,
            output_dir=self.tmpdir,
        )

        dm = self.build_count_datamodule()
        batch = next(iter(dm.train_dataloader()))

        outputs, _ = model.forward(batch)
        t = list(outputs.keys())[0]

        self.assertEqual(
            outputs[t]["count_mean"].shape,
            (self.batch_size, self.num_genes),
        )
        
    def test_checkpoint_created(self):
        self.assertTrue(os.path.exists(self.count_ckpt_path))

class TestGeneration(TestCountDecoder):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # builds BOTH checkpoints

        cls.context_tps = [1, 2]

    def _build_generation_model(self):
        def _factory():
            return CountDecoderTrainer(
                **self.countdecoder_config,
                ckpt_masking_path=self.ckpt_path,
                ckpt_count_path=self.count_ckpt_path,  # ✅ reuse trained ckpt
                output_dir=self.tmpdir,
                context_tps=self.context_tps,
                generate=True,
                temperature=1.5,
                iterations=19,
            )
        return _factory()

    def _build_generation_datamodule(self):
        dm = PerturbGenDataModule(
            src_counts=self.src_counts,
            tgt_counts_dict=self.tgt_counts_dict,
            src_dataset=self.src_dataset,
            tgt_datasets=self.tgt_datasets,
            batch_size=self.batch_size,
            num_workers=1,
            pred_tps=self.pred_tps,
            context_tps=self.context_tps,
            n_total_tps=self.n_total_tps,
            test_indices=np.arange(20),
            max_len=self.max_seq_length,
            condition_keys=self.condition_keys,
            condition_encodings={"tmp_batch": {1.0: 0}},
            conditions=self.conditions,
            conditions_combined=self.conditions_combined,
            use_weighted_sampler=False,
        )
        dm.setup()
        return dm

    def test_generation(self):
        model = self._build_generation_model()
        dm = self._build_generation_datamodule()

        trainer = pl.Trainer(
            limit_test_batches=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.test(model, dm)


# =========================
# Checkpoint Reproducibility Test
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_MASK_CKPT = f'{BASE_DIR}/res/reference_mask.ckpt'
NEW_MASK_CKPT = f'{BASE_DIR}/res/new_mask.ckpt'
REF_COUNT_CKPT = f'{BASE_DIR}/res/reference_count.ckpt'
NEW_COUNT_CKPT = f'{BASE_DIR}/res/new_count.ckpt'
checkpoint_pairs = {
    "masking": (REF_MASK_CKPT, NEW_MASK_CKPT),
    "count": (REF_COUNT_CKPT, NEW_COUNT_CKPT),
    # Add more pairs as needed
}

def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

def compare_weights(ref_sd, new_sd, atol=1e-5, rtol=1e-4):
    mismatches = []
    max_diff = 0.0

    for k in ref_sd:
        if k not in new_sd:
            mismatches.append((k, "missing in new"))
            continue

        ref = ref_sd[k]
        new = new_sd[k]

        if not torch.allclose(ref, new, atol=atol, rtol=rtol):
            diff = (ref - new).abs().max().item()
            max_diff = max(max_diff, diff)
            mismatches.append((k, diff))

    return mismatches, max_diff

class TestCheckpointConsistency(unittest.TestCase):

    def test_checkpoint_match(self):
        paths = [REF_MASK_CKPT, NEW_MASK_CKPT, REF_COUNT_CKPT, NEW_COUNT_CKPT]
        for path in paths:
            if not os.path.exists(path):
                raise Warning(f"Checkpoint {path} not found. Skipping consistency test.")
            else:
                pass    
        failures = []

        if len(failures) > 0:
            msg = "\n".join([
                f"{name}: {n} mismatches (max diff {d})"
                for name, n, d in failures
            ])
            self.fail(f"Checkpoint mismatches found:\n{msg}")
        else:
            for name, (ref_path, new_path) in checkpoint_pairs.items():
                print(f"\n--- Testing {name} checkpoint ---")

                ref_path = os.path.abspath(ref_path)
                new_path = os.path.abspath(new_path)

                self.assertTrue(os.path.exists(ref_path), f"Missing {ref_path}")
                self.assertTrue(os.path.exists(new_path), f"Missing {new_path}")

                ref_sd = load_state_dict(ref_path)
                new_sd = load_state_dict(new_path)

                mismatches, max_diff = compare_weights(ref_sd, new_sd)

                if len(mismatches) > 0:
                    print(f"[{name}] ❌ {len(mismatches)} mismatches (max diff: {max_diff})")
                    for k, d in mismatches[:5]:
                        print(f" - {k}: {d}")
                    failures.append((name, len(mismatches), max_diff))
                else:
                    print(f"[{name}] ✅ match")

        # Fail once at the end (so all checkpoints are checked)
        if failures:
            msg = "\n".join([
                f"{name}: {n} mismatches (max diff {d})"
                for name, n, d in failures
            ])
            self.fail(f"Checkpoint mismatches found:\n{msg}")

if __name__ == "__main__":
    unittest.main()
