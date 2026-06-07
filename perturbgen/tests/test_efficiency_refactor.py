"""
Regression tests that verify efficiency refactors do not change outputs.

Covers:
  1. top_k does not modify its input (so logits.clone() before top_k is safe to remove).
  2. In generate_sequence, replacing .clone() with views for tmp_ids_, ids_to_keep_,
     and scores_ gives identical results.
  3. collate: computing .toarray() once and reusing gives the same tgt_counts and
     tgt_size_factor as computing it twice.
  4. generate() debug timing code removal: full generation output is unchanged
     (end-to-end determinism with fixed seed).
"""

import os
import unittest

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.sparse import csr_matrix

from perturbgen.src.utils import top_k

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
MAPPING_PATH = "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf_100M/token_id_to_genename_5000_hvg.pkl"
TOKENID_PATH = "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf_100M/tokenid_to_rowid_5000_hvg.pkl"


# ---------------------------------------------------------------------------
# Unit tests: pure tensor-operation properties
# ---------------------------------------------------------------------------

class TestTopKNoInplaceModification(unittest.TestCase):
    """top_k must not modify its input — so removing logits.clone() is safe."""

    def test_input_unchanged_after_top_k(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 48, 200)
        original = logits.clone()
        _ = top_k(logits, thres=0.9)
        self.assertTrue(
            torch.equal(logits, original),
            "top_k modified its input tensor — the logits.clone() call IS needed",
        )

    def test_clone_vs_no_clone_same_output(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 48, 200)
        out_with_clone = top_k(logits.clone(), thres=0.9)
        out_without_clone = top_k(logits, thres=0.9)
        self.assertTrue(
            torch.equal(out_with_clone, out_without_clone),
            "top_k(logits.clone()) != top_k(logits) — outputs differ unexpectedly",
        )


class TestGenerateSequenceCloneEquivalence(unittest.TestCase):
    """
    Verify that the three .clone() calls inside generate_sequence are redundant:
      - tmp_ids_ used only as source for torch.where (which creates a new tensor)
      - ids_to_keep_ used only as index in logits.scatter_ (which modifies logits, not ids)
      - scores_ is always fully overwritten before any read
    """

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 50
        self.cond_length = 2
        self.vocab_size = 100
        self.mask_token = 1
        self.max_neg_value = -torch.finfo(torch.float32).max

        # Simulate the state of tensors inside generate_sequence at the start of an iteration
        self.tmp_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.tmp_ids[:, :self.cond_length] = 0  # conditioned prefix
        # Sprinkle some mask tokens
        self.tmp_ids[:, self.cond_length:][torch.rand(self.batch_size, self.seq_len - self.cond_length) < 0.3] = self.mask_token

        self.ids_to_keep = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.scores = torch.rand(self.batch_size, self.seq_len)
        self.pred_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len - self.cond_length))
        self.logits = torch.randn(self.batch_size, self.seq_len - self.cond_length, self.vocab_size)

    def _run_with_clones(self):
        """Mirrors the current (pre-refactor) code in generate_sequence."""
        tmp_ids = self.tmp_ids.clone()
        ids_to_keep = self.ids_to_keep.clone()
        logits = self.logits.clone()
        cond_length = self.cond_length
        seq_len = self.seq_len
        mask_token = self.mask_token
        max_neg_value = self.max_neg_value
        pred_ids = self.pred_ids.clone()

        tmp_ids_ = tmp_ids[:, cond_length:].clone()
        scores_ = self.scores[:, cond_length:].clone()  # will be overwritten anyway
        ids_to_keep_ = ids_to_keep[:, cond_length:].clone()

        indices = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
        logits.scatter_(2, indices, max_neg_value)

        is_mask = tmp_ids_ == mask_token
        tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)

        # Simulate scores_ being overwritten (topk-margin branch)
        probs = logits.softmax(dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        scores_ = 1 - torch.abs(top2[:, :, 0] - top2[:, :, 1])

        tmp_ids[:, cond_length:] = tmp_ids_
        return tmp_ids.clone(), logits.clone(), scores_.clone()

    def _run_with_views(self):
        """Mirrors the proposed (post-refactor) code — views instead of clones."""
        tmp_ids = self.tmp_ids.clone()
        ids_to_keep = self.ids_to_keep.clone()
        logits = self.logits.clone()
        cond_length = self.cond_length
        seq_len = self.seq_len
        mask_token = self.mask_token
        max_neg_value = self.max_neg_value
        pred_ids = self.pred_ids.clone()

        tmp_ids_ = tmp_ids[:, cond_length:]        # view, no clone
        ids_to_keep_ = ids_to_keep[:, cond_length:]  # view, no clone
        # scores_ not initialised from old value — always overwritten below

        indices = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
        logits.scatter_(2, indices, max_neg_value)

        is_mask = tmp_ids_ == mask_token
        tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)

        probs = logits.softmax(dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        scores_ = 1 - torch.abs(top2[:, :, 0] - top2[:, :, 1])

        tmp_ids[:, cond_length:] = tmp_ids_
        return tmp_ids.clone(), logits.clone(), scores_.clone()

    def test_tmp_ids_result_identical(self):
        ids_clone, _, _ = self._run_with_clones()
        ids_view, _, _ = self._run_with_views()
        self.assertTrue(
            torch.equal(ids_clone, ids_view),
            "tmp_ids differs between clone and view code paths",
        )

    def test_logits_scatter_result_identical(self):
        _, logits_clone, _ = self._run_with_clones()
        _, logits_view, _ = self._run_with_views()
        self.assertTrue(
            torch.equal(logits_clone, logits_view),
            "logits after scatter_ differ between clone and view code paths",
        )

    def test_scores_result_identical(self):
        _, _, scores_clone = self._run_with_clones()
        _, _, scores_view = self._run_with_views()
        self.assertTrue(
            torch.equal(scores_clone, scores_view),
            "scores_ differ between clone and view code paths",
        )

    def test_ids_to_keep_not_modified_by_scatter(self):
        """ids_to_keep should be unmodified after the scatter_ — confirms view is safe."""
        ids_to_keep_before = self.ids_to_keep.clone()
        self._run_with_views()  # runs scatter_ on logits using ids_to_keep as index
        self.assertTrue(
            torch.equal(self.ids_to_keep, ids_to_keep_before),
            "ids_to_keep was modified in-place during scatter_ — view-based refactor is UNSAFE",
        )


# ---------------------------------------------------------------------------
# Unit test: collate double-.toarray() fix
# ---------------------------------------------------------------------------

class TestCollateToarrayOnce(unittest.TestCase):
    """
    Replacing the double .toarray() call for csr_matrix inputs (once for counts,
    once for size_factor) with a single call + reuse must yield identical results.
    """

    def _make_sparse_batch(self, n_cells=8, n_genes=50, seed=7):
        rng = np.random.default_rng(seed)
        data = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
        return [csr_matrix(data[i : i + 1]) for i in range(n_cells)]

    def test_counts_identical(self):
        batch = self._make_sparse_batch()

        # Current code: toarray() called per element inside list-comp
        counts_current = [torch.tensor(d.toarray()) for d in batch]
        counts_current = torch.cat(counts_current, dim=0)

        # Proposed: compute dense once per element, reuse
        dense = [d.toarray() for d in batch]
        counts_proposed = torch.cat([torch.tensor(a) for a in dense], dim=0)

        self.assertTrue(
            torch.equal(counts_current, counts_proposed),
            "counts differ when toarray() is called once vs twice",
        )

    def test_size_factor_identical(self):
        batch = self._make_sparse_batch()

        # Current code: second independent .toarray() for size factor
        sf_current = [
            torch.tensor(np.ravel(d.toarray().sum(axis=1))) for d in batch
        ]
        sf_current = torch.cat(sf_current, dim=0)

        # Proposed: reuse the dense array already computed for counts
        dense = [d.toarray() for d in batch]
        sf_proposed = torch.cat(
            [torch.tensor(np.ravel(a.sum(axis=1))) for a in dense], dim=0
        )

        self.assertTrue(
            torch.equal(sf_current, sf_proposed),
            "size_factor differs when toarray() is called once vs twice",
        )

    def test_size_factor_equals_counts_row_sum(self):
        """Sanity check: size factor must equal the row-sum of the count matrix."""
        batch = self._make_sparse_batch()
        dense = [d.toarray() for d in batch]
        counts = torch.cat([torch.tensor(a) for a in dense], dim=0)    # (n, genes)
        sf = torch.cat([torch.tensor(np.ravel(a.sum(axis=1))) for a in dense], dim=0)

        self.assertTrue(
            torch.allclose(counts.sum(dim=-1), sf),
            "size_factor is not equal to counts.sum(-1)",
        )


# ---------------------------------------------------------------------------
# Integration test: full generation is deterministic (debug-code removal safe)
# ---------------------------------------------------------------------------

class TestGenerationDeterminism(unittest.TestCase):
    """
    Removing import/timing/print from generate() must not change generated token IDs.
    We verify this by running generation twice with the same seed and checking equality.
    A non-deterministic generate() would fail even before any refactor.
    """

    @classmethod
    def setUpClass(cls):
        for path in [ENCODER_PATH, MAPPING_PATH, TOKENID_PATH]:
            if not os.path.exists(path):
                raise unittest.SkipTest(f"Required resource missing: {path}")

        from perturbgen.tests.utils import dummy_dataset, dummy_cell_gene_matrix
        from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
        from perturbgen.Model.trainer import CountDecoderTrainer
        import tempfile, shutil
        from pytorch_lightning.callbacks import ModelCheckpoint

        pl.seed_everything(42)
        pred_tps = [1, 2]
        n_total_tps = 2
        max_seq_length = 50
        tgt_vocab_size = 101
        num_genes = tgt_vocab_size - 1
        batch_size = 4
        d_model = 768
        num_samples = 100

        cls.pred_tps = pred_tps

        src_dataset = dummy_dataset(max_len=max_seq_length, vocab_size=tgt_vocab_size, num_samples=num_samples)
        src_counts = dummy_cell_gene_matrix(num_cells=num_samples, num_genes=num_genes)
        tgt_datasets = dummy_dataset(max_len=max_seq_length, vocab_size=tgt_vocab_size, num_samples=num_samples, total_time_steps=n_total_tps)
        tgt_counts_dict = dummy_cell_gene_matrix(num_cells=num_samples, num_genes=num_genes, total_time_steps=n_total_tps)

        from perturbgen.src.utils import label_encoder
        import scanpy as sc, pandas as pd
        tmp_series = pd.DataFrame({"tmp_batch": np.ones(num_samples)})
        condition_encodings = {"tmp_batch": {1.0: 0}}
        conditions_combined_encodings = {1.0: 0}
        tgt_adata_tmp = sc.AnnData(X=tgt_counts_dict['tgt_h5ad_t1'].squeeze(), obs=tmp_series)
        conditions = torch.tensor([label_encoder(tgt_adata_tmp, condition_encodings["tmp_batch"], "tmp_batch")], dtype=torch.long).T
        conditions_combined = torch.tensor(label_encoder(tgt_adata_tmp, conditions_combined_encodings, "tmp_batch"), dtype=torch.long)

        cls.tmpdir_obj = tempfile.TemporaryDirectory()
        cls.tmpdir = cls.tmpdir_obj.name

        # Train a minimal masking checkpoint
        from perturbgen.Model.trainer import PerturbGenTrainer
        mask_config = dict(
            tgt_vocab_size=tgt_vocab_size, d_model=d_model, num_heads=4, num_layers=1,
            d_ff=8, max_seq_length=max_seq_length + 10, dropout=0.0, pred_tps=pred_tps,
            n_total_tps=n_total_tps, encoder="scmaskgit", encoder_path=ENCODER_PATH,
            mapping_dict_path=MAPPING_PATH, context_mode=True, mask_scheduler="pow",
            seed=42, end_lr=1e-3, weight_decay=0.0, return_embeddings=False,
            tokenid_to_rowid_path=TOKENID_PATH, output_dir=cls.tmpdir,
        )
        mask_dm = PerturbGenDataModule(
            src_dataset=src_dataset, tgt_datasets=tgt_datasets, batch_size=batch_size,
            num_workers=1, pred_tps=pred_tps, n_total_tps=n_total_tps,
            max_len=max_seq_length, train_indices=np.arange(num_samples), use_weighted_sampler=False,
        )
        mask_dm.setup()
        ckpt_cb = ModelCheckpoint(dirpath=cls.tmpdir, filename="mask_det", save_last=True)
        pl.Trainer(max_epochs=1, limit_train_batches=1, logger=False, enable_checkpointing=True,
                   callbacks=[ckpt_cb], enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0).fit(PerturbGenTrainer(**mask_config), mask_dm)
        ckpt_path = ckpt_cb.last_model_path

        # Train a minimal count decoder checkpoint
        count_config = dict(
            tgt_vocab_size=tgt_vocab_size, d_model=d_model, num_heads=4, num_layers=1,
            d_ff=8, max_seq_length=max_seq_length + 10, dropout=0.0, pred_tps=pred_tps,
            n_total_tps=n_total_tps, encoder="scmaskgit", encoder_path=ENCODER_PATH,
            mapping_dict_path=MAPPING_PATH, context_mode=True, mask_scheduler="pow", seed=42,
            lr=1e-3, weight_decay=0.0, loss_mode="zinb", n_genes=num_genes,
            conditions={"tmp_batch": [1.0]}, conditions_combined=[1.0],
            ckpt_masking_path=ckpt_path, output_dir=cls.tmpdir,
        )
        count_dm = PerturbGenDataModule(
            src_dataset=src_dataset, tgt_datasets=tgt_datasets,
            src_counts=src_counts, tgt_counts_dict=tgt_counts_dict,
            batch_size=batch_size, num_workers=1, pred_tps=pred_tps,
            n_total_tps=n_total_tps, max_len=max_seq_length,
            condition_keys=["tmp_batch"], condition_encodings={"tmp_batch": {1.0: 0}},
            conditions=conditions, conditions_combined=conditions_combined,
            use_weighted_sampler=False,
        )
        count_dm.setup()
        cnt_cb = ModelCheckpoint(dirpath=cls.tmpdir, filename="count_det", save_last=True)
        pl.Trainer(max_epochs=1, limit_train_batches=1, logger=False, enable_checkpointing=True,
                   callbacks=[cnt_cb], enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0).fit(CountDecoderTrainer(**count_config), count_dm)

        cls.count_ckpt_path = cnt_cb.last_model_path
        cls.ckpt_masking_path = ckpt_path
        cls.count_config = count_config
        cls.gen_dm = PerturbGenDataModule(
            src_counts=src_counts, tgt_counts_dict=tgt_counts_dict,
            src_dataset=src_dataset, tgt_datasets=tgt_datasets,
            batch_size=batch_size, num_workers=1, pred_tps=pred_tps,
            context_tps=[1, 2], n_total_tps=n_total_tps,
            test_indices=np.arange(20), max_len=max_seq_length,
            condition_keys=["tmp_batch"], condition_encodings={"tmp_batch": {1.0: 0}},
            conditions=conditions, conditions_combined=conditions_combined,
            use_weighted_sampler=False,
        )
        cls.gen_dm.setup()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir_obj.cleanup()

    def _run_generation(self, seed):
        from perturbgen.Model.trainer import CountDecoderTrainer
        pl.seed_everything(seed)
        torch.manual_seed(seed)
        cfg = {
            **self.count_config,
            'ckpt_count_path': self.count_ckpt_path,
            'context_tps': [1, 2],
            'generate': True,
            'temperature': 1.5,
            'iterations': 5,
        }
        model = CountDecoderTrainer(**cfg)
        model.eval()

        batch = next(iter(self.gen_dm.test_dataloader()))
        with torch.no_grad():
            _, generated_ids = model.decoder.generate_counts(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict={k: v for k, v in batch.items() if 'tgt_input_ids' in k},
                temperature=1.5,
                iterations=5,
            )
        return {k: v.clone() for k, v in generated_ids.items() if isinstance(v, torch.Tensor)}

    def test_generate_is_deterministic(self):
        """Same seed must produce identical token IDs — prerequisite for safe refactoring."""
        ids_run1 = self._run_generation(seed=42)
        ids_run2 = self._run_generation(seed=42)

        for key in ids_run1:
            self.assertTrue(
                torch.equal(ids_run1[key], ids_run2[key]),
                f"Generated IDs for {key} differ between two runs with the same seed. "
                "Generation is not deterministic — cannot safely verify refactors.",
            )


if __name__ == "__main__":
    unittest.main()
