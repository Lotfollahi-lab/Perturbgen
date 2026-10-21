"""Clean, scvi-tools-style Python API for PerturbGen.

This wraps the existing training / perturbation entry points so a notebook can
do::

    import perturbgen as pg

    model = pg.PerturbGen(
        "T_perturb/tokenized_data/lps_90min_perturb",
        source="90m_LPS",
        encoder_path="pretraining_cohort/....ckpt",
        var_list=["cell_type_harmonized", "cell_pairing_index", "time_after_LPS"],
        pred_tps=[1, 2],
    )
    masking_ckpt = model.train_masking(output_dir="lps_model", max_epochs=20, batch_size=64)
    count_ckpt = model.train_count(output_dir="lps_model", masking_ckpt=masking_ckpt)
    model.save("lps_model")                       # or: pg.PerturbGen.load("lps_model")

    pred = model.perturb(genes=["ENSG00000125538"],   # IL1B
                         mode="knockout", where="source")

Tokenization is a separate step (:func:`tokenize`, used in the train_and_perturb tutorial); this API starts
from the tokenized-data directory it produces. Everything runs **in process** —
no shelling out to CLI scripts.

Requires a CUDA GPU for `train_masking()`/`train_count()` and `perturb()` (same as the CLI).
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


@contextlib.contextmanager
def _preserve_cwd():
    """Restore the working directory after a block.

    The underlying train/val entry points call ``os.chdir(ROOT)`` at import, which
    would otherwise silently move the notebook's cwd.
    """
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)


# perturbation-mode aliases -> the underlying `perturbation_mode` value
_MODE_ALIASES = {
    "knockout": "mask",
    "mask": "mask",
    "overexpress": "overexpress",
    "delete": "delete",
    "pad": "pad",
}
# where the edit is applied -> `perturbation_sequence` value
_WHERE_ALIASES = {"source": "src", "src": "src", "target": "tgt", "tgt": "tgt"}


def tokenize(
    adata_path: str,
    dataset: str,
    reference_time: str,
    *,
    time_obs: str,
    time_point_order: Sequence[str],
    var_list: Sequence[str],
    main_pairing_obs: str,
    gene_median_path: str,
    token_dict_path: str,
    gene_mapping_path: str,
    n_hvg: int = 2000,
    gene_filtering_mode: str = "hvg",
    hvg_mode: str = "after_tokenisation",
    pairing_mode: str = "stratified",
    nproc: int = 8,
    exclude_non_gf_genes: bool = True,
    cell_gene_filter: bool = False,
    remove_mito_ribo_genes: bool = False,
    hvg_flavor: str = "seurat_v3",
    opt_pairing_obs: Optional[Sequence[str]] = None,
    pairing_file: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Tokenize + pair an AnnData; returns the tokenized-data directory.

    Idempotent: if the output already exists it is returned as-is without
    re-tokenizing (pass ``overwrite=True`` to force a rebuild), so re-running
    the cell is a fast no-op.

    A standalone preprocessing step (no model needed) — use as ``pg.tokenize(...)``;
    the returned path is what you pass to :class:`PerturbGen`.

    `reference_time` is the source state (source files are written as
    ``<reference_time>.dataset`` / ``.h5ad``); pairs run source -> later
    `time_point_order` entries.

    `hvg_mode`: ``after_tokenisation`` selects `n_hvg` genes from the
    Geneformer-tokenizable set (yields exactly `n_hvg`); ``before_tokenisation``
    selects `n_hvg` HVGs first and then intersects with the token vocabulary
    (yields fewer).
    """
    from perturbgen.configs.paths import TOKENIZED_DIR

    out_dir = os.path.join(str(TOKENIZED_DIR), dataset)
    if not overwrite and glob.glob(os.path.join(out_dir, f"dataset_{n_hvg}_hvg_src")):
        print(f"[PerturbGen] tokenized dataset already exists — skipping (overwrite=True to redo): {out_dir}")
        return out_dir

    argv = [
        "--h5ad_path", os.path.abspath(adata_path),
        "--dataset", dataset,
        "--gene_filtering_mode", gene_filtering_mode,
        "--exclude_non_GF_genes", str(exclude_non_gf_genes),
        "--cell_gene_filter", str(cell_gene_filter),
        "--remove_mito_ribo_genes", str(remove_mito_ribo_genes),
        "--hvg_flavor", hvg_flavor,
        "--hvg_mode", hvg_mode,
        "--var_list", *list(var_list),
        "--pairing_mode", pairing_mode,
        "--time_obs", time_obs,
        "--main_pairing_obs", main_pairing_obs,
        "--nproc", str(nproc),
        "--n_hvg", str(n_hvg),
        "--reference_time", reference_time,
        "--time_point_order", *list(time_point_order),
        "--gene_median_path", os.path.abspath(gene_median_path),
        "--token_dict_path", os.path.abspath(token_dict_path),
        "--gene_mapping_path", os.path.abspath(gene_mapping_path),
    ]
    if pairing_mode == "mapping" and pairing_file:
        argv += ["--pairing_file", os.path.abspath(pairing_file)]
    if opt_pairing_obs:
        argv += ["--opt_pairing_obs", *list(opt_pairing_obs)]

    print(f"[PerturbGen] tokenizing {adata_path} -> dataset '{dataset}' ...")
    with _preserve_cwd():
        from perturbgen.pp.GF_tokenisation import main as _tok_main
        _tok_main(argv)
    return out_dir


class PerturbGen:
    """A trained (or trainable) PerturbGen model over one tokenized dataset."""

    def __init__(
        self,
        tokenized_dir: str,
        *,
        source: str,
        n_hvg: Optional[int] = None,
        encoder_path: Optional[str] = None,
        encoder: str = "scmaskgit",
        n_layers: int = 6,
        d_ff: int = 64,
        d_model: int = 768,
        pred_tps: Sequence[int] = (1, 2),
        var_list: Sequence[str] = ("cell_type_harmonized", "cell_pairing_index", "time_after_LPS"),
        context_mode: bool = True,
        cond_list: Optional[Sequence[str]] = None,
        mask_scheduler: str = "pow",
        pos_encoding_mode: str = "time_pos_sin",
        loss_mode: str = "zinb",
        seed: int = 42,
        max_len: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
    ) -> None:
        """Build a model over a tokenized dataset.

        `tokenized_dir` is the output of :func:`tokenize` (or a released tokenized
        dataset); `source` is the reference/source time point (e.g. "90m_LPS",
        "normal"). File paths are inferred from the standard tokenizer layout::

            <tokenized_dir>/dataset_<n_hvg>_hvg_src/<source>.dataset
            <tokenized_dir>/dataset_<n_hvg>_hvg_tgt/
            <tokenized_dir>/h5ad_pairing_<n_hvg>_hvg_src/<source>.h5ad
            <tokenized_dir>/h5ad_pairing_<n_hvg>_hvg_tgt/
            <tokenized_dir>/token_id_to_genename_<n_hvg>_hvg.pkl
            <tokenized_dir>/tokenid_to_rowid_<n_hvg>_hvg.pkl

        `n_hvg` is only the ``<n_hvg>_hvg`` filename key; leave it ``None`` (default)
        to auto-detect it from the tokenized dir, or pass it to disambiguate when
        several variants live side by side.

        `max_len` / `tgt_vocab_size` fix the model's sequence length and target
        vocabulary size (base values, before the model's offsets). Leave both
        ``None`` (default) to auto-detect them from the data; set both to match a
        model trained with explicit values (the released LPS models: 797 / 2002).
        They are used for training, embedding extraction and perturbation alike.
        """
        if (max_len is None) != (tgt_vocab_size is None):
            raise ValueError("Set both max_len and tgt_vocab_size, or neither (auto-detect).")
        self.tokenized_dir = os.path.abspath(tokenized_dir)
        self.source = source
        # n_hvg is only a filename key (dataset_<n_hvg>_hvg_*); auto-detect it from
        # the tokenized dir so it can't be mismatched with the tokenization.
        if n_hvg is None:
            n_hvg = _detect_n_hvg(self.tokenized_dir)
        self.n_hvg = n_hvg
        d = Path(self.tokenized_dir)
        sfx = f"{n_hvg}_hvg"
        # absolute so the entry points' os.chdir(ROOT) can't break them
        self.paths = {
            "src_dataset": str(d / f"dataset_{sfx}_src" / f"{source}.dataset"),
            "tgt_dataset_folder": str(d / f"dataset_{sfx}_tgt"),
            "src_adata": str(d / f"h5ad_pairing_{sfx}_src" / f"{source}.h5ad"),
            "tgt_adata_folder": str(d / f"h5ad_pairing_{sfx}_tgt"),
            "mapping_dict_path": str(d / f"token_id_to_genename_{sfx}.pkl"),
            "tokenid_to_rowid_path": str(d / f"tokenid_to_rowid_{sfx}.pkl"),
        }
        for key, pth in self.paths.items():
            if not os.path.exists(pth):
                raise FileNotFoundError(
                    f"Expected tokenized file for '{key}' not found: {pth}\n"
                    f"Check tokenized_dir / source / n_hvg."
                )
        self.hparams = dict(
            encoder_path=os.path.abspath(encoder_path) if encoder_path else None,
            encoder=encoder, n_layers=n_layers,
            d_ff=d_ff, d_model=d_model, pred_tps=list(pred_tps),
            var_list=list(var_list), context_mode=context_mode,
            cond_list=list(cond_list) if cond_list else None,
            mask_scheduler=mask_scheduler, pos_encoding_mode=pos_encoding_mode,
            loss_mode=loss_mode, seed=seed,
            max_len=max_len, tgt_vocab_size=tgt_vocab_size,
        )
        # populated by train()/load()
        self.masking_ckpt: Optional[str] = None
        self.count_ckpt: Optional[str] = None

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _common_argv(self, batch_size: int, n_workers: int, num_node: int,
                     seed: Optional[int] = None, compile: bool = True) -> List[str]:
        h, p = self.hparams, self.paths
        seed = h["seed"] if seed is None else seed
        argv = [
            "--split", "False", "--splitting_mode", "stratified",
            "--src_dataset", p["src_dataset"],
            "--tgt_dataset_folder", p["tgt_dataset_folder"],
            "--src_adata", p["src_adata"],
            "--tgt_adata_folder", p["tgt_adata_folder"],
            "--mapping_dict_path", p["mapping_dict_path"],
            "--batch_size", str(batch_size),
            "--n_workers", str(n_workers),
            "--num_layers", str(h["n_layers"]),
            "--d_ff", str(h["d_ff"]),
            "--d_model", str(h["d_model"]),
            "--pred_tps", *[str(t) for t in h["pred_tps"]],
            "--var_list", *h["var_list"],
            "--encoder", h["encoder"],
            "--context_mode", str(h["context_mode"]),
            "--pos_encoding_mode", h["pos_encoding_mode"],
            "--mask_scheduler", h["mask_scheduler"],
            "--num_node", str(num_node),
            "--seed", str(seed),
            "--single_device", "True",   # in-process/notebook: single GPU, no DDP
            "--wandb_mode", "disabled",  # WandbLogger.experiment -> wandb.init() hangs in a headless kernel
            "--compile", str(compile),   # release masking used compile=True; off = snappier start, tiny FP drift
        ]
        if h.get("cond_list"):
            argv += ["--cond_list", *h["cond_list"]]
        if h["encoder_path"]:
            argv += ["--encoder_path", h["encoder_path"]]
        argv += self._dims_argv()
        return argv

    def _dims_argv(self) -> List[str]:
        h = self.hparams
        if h.get("max_len") is None:
            return []
        return ["--max_len", str(h["max_len"]), "--tgt_vocab_size", str(h["tgt_vocab_size"])]

    def train_masking(
        self,
        output_dir: str,
        *,
        max_epochs: int = 20,
        batch_size: int = 64,
        cellgen_lr: float = 1e-4,
        cellgen_wd: float = 1e-4,
        n_workers: int = 4,
        num_node: int = 1,
        use_weighted_sampler: bool = False,
        ckpt_every_n_epochs: int = 1,
        seed: Optional[int] = None,
        compile: bool = True,
    ) -> str:
        """Train the masking model (step 1 of 2).

        A checkpoint is saved for every epoch under
        ``<output_dir>/masking/checkpoints`` — inspect the training curves (e.g.
        in Weights & Biases) to decide which epoch to feed the count decoder.
        Sets and returns the **metric-best** checkpoint, which ``train_count``
        uses by default; override it there to pick a different epoch.
        """
        mask_out = os.path.join(os.path.abspath(output_dir), "masking")
        argv = [
            "--train_mode", "masking", "--split_obs", "cell_type_harmonized",
            "--output_dir", mask_out,
            "--epochs", str(max_epochs),
            "--cellgen_lr", str(cellgen_lr), "--cellgen_wd", str(cellgen_wd),
            "--use_weighted_sampler", str(use_weighted_sampler),
            "--ckpt_every_n_epochs", str(ckpt_every_n_epochs),
        ] + self._common_argv(batch_size, n_workers, num_node, seed=seed, compile=compile)
        print("[PerturbGen] training masking model ...")
        with _preserve_cwd():
            from perturbgen.train import main as _train_main  # in-process, no subprocess
            best = _train_main(argv)
        self.masking_ckpt = best or _latest_ckpt(mask_out)
        n_ckpts = len(glob.glob(os.path.join(mask_out, "checkpoints", "*.ckpt")))
        print(f"[PerturbGen] {n_ckpts} masking checkpoint(s) under {mask_out}/checkpoints")
        print(f"[PerturbGen] metric-best (default for train_count): {self.masking_ckpt}")
        return self.masking_ckpt

    def train_count(
        self,
        output_dir: str,
        *,
        masking_ckpt: Optional[str] = None,
        max_epochs: int = 20,
        batch_size: int = 64,
        count_lr: float = 1e-3,
        count_wd: float = 1e-4,
        cellgen_lr: float = 1e-4,
        cellgen_wd: float = 1e-4,
        count_dropout: float = 0.0,
        mlm_prob: float = 0.15,
        n_workers: int = 4,
        num_node: int = 1,
        ckpt_every_n_epochs: int = 1,
        seed: Optional[int] = None,
        compile: bool = True,
    ) -> str:
        """Train the count decoder from a masking checkpoint (step 2 of 2).

        `masking_ckpt` defaults to the metric-best from ``train_masking``; pass an
        explicit checkpoint (e.g. a specific epoch under
        ``<output_dir>/masking/checkpoints``) to use a different one. Returns the
        trained count-decoder checkpoint.
        """
        ckpt = masking_ckpt or self.masking_ckpt
        if ckpt is None:
            raise ValueError(
                "train_count needs a masking checkpoint; run train_masking() first "
                "or pass masking_ckpt=<path>."
            )
        ckpt = os.path.abspath(ckpt)
        count_out = os.path.join(os.path.abspath(output_dir), "count")
        h = self.hparams
        argv = [
            "--train_mode", "count", "--output_dir", count_out,
            "--ckpt_masking_path", ckpt,
            "--epochs", str(max_epochs),
            "--count_lr", str(count_lr), "--count_wd", str(count_wd),
            "--cellgen_lr", str(cellgen_lr), "--cellgen_wd", str(cellgen_wd),
            "--mlm_prob", str(mlm_prob), "--count_dropout", str(count_dropout),
            "--loss_mode", h["loss_mode"],
            "--ckpt_every_n_epochs", str(ckpt_every_n_epochs),
            "--use_positional_encoding", "False", "--layer_norm", "False",
        ] + self._common_argv(batch_size, n_workers, num_node, seed=seed, compile=compile)
        print(f"[PerturbGen] training count decoder from {ckpt} ...")
        with _preserve_cwd():
            from perturbgen.train import main as _train_main  # in-process, no subprocess
            best = _train_main(argv)
        self.count_ckpt = best or _latest_ckpt(count_out)
        return self.count_ckpt

    # ------------------------------------------------------------------ #
    # embeddings
    # ------------------------------------------------------------------ #
    def get_embeddings(
        self,
        output_dir: str = "embeddings",
        *,
        masking_ckpt: Optional[str] = None,
        return_gene_embs: bool = True,
        return_cell_embs: bool = True,
        gene_embs_condition: Optional[str] = None,
        batch_size: int = 64,
        n_workers: int = 4,
    ) -> str:
        """Extract gene/cell embeddings from the trained masking model.

        Writes embedding AnnData(s) under `output_dir` and returns that path.
        Requires a trained masking checkpoint (run train_masking()
        first, or pass `masking_ckpt=`).
        """
        ckpt = masking_ckpt or self.masking_ckpt
        if ckpt is None:
            raise ValueError(
                "get_embeddings needs a masking checkpoint; run train_masking() "
                "first or pass masking_ckpt=."
            )
        if return_gene_embs and not gene_embs_condition:
            raise ValueError(
                "return_gene_embs=True requires gene_embs_condition=<obs column>: gene "
                "embeddings are extracted per unique value of that column (e.g. "
                "gene_embs_condition='time_after_LPS'). Pass one, or set "
                "return_gene_embs=False to get cell embeddings only."
            )
        h, p = self.hparams, self.paths
        output_dir = os.path.abspath(output_dir)
        ckpt = os.path.abspath(ckpt)
        os.makedirs(output_dir, exist_ok=True)
        argv = [
            "--test_mode", "masking", "--split", "False", "--splitting_mode", "stratified",
            "--return_embeddings", str(return_cell_embs),
            "--return_attn", "False",
            "--single_device", "True",   # in-process/notebook: single GPU, no DDP
            "--wandb_mode", "disabled",  # avoid wandb.init() hang in a headless kernel
            "--generate", "False",
            "--return_gene_embs", str(return_gene_embs),
            "--ckpt_masking_path", ckpt,
            "--output_dir", output_dir,
            "--src_dataset", p["src_dataset"],
            "--tgt_dataset_folder", p["tgt_dataset_folder"],
            "--src_adata", p["src_adata"],
            "--tgt_adata_folder", p["tgt_adata_folder"],
            "--mapping_dict_path", p["mapping_dict_path"],
            "--tokenid_to_rowid_path", p["tokenid_to_rowid_path"],
            "--batch_size", str(batch_size),
            "--n_workers", str(n_workers),
            "--d_ff", str(h["d_ff"]),
            "--num_layers", str(h["n_layers"]),
            "--d_model", str(h["d_model"]),
            "--pred_tps", *[str(t) for t in h["pred_tps"]],
            "--var_list", *h["var_list"],
            *(["--cond_list", *h["cond_list"]] if h.get("cond_list") else []),
            "--encoder", h["encoder"],
            "--context_mode", str(h["context_mode"]),
            "--mask_scheduler", h["mask_scheduler"],
            "--pos_encoding_mode", h["pos_encoding_mode"],
            # training-only args carried through for argparse compatibility
            "--cellgen_lr", "1e-4", "--cellgen_wd", "1e-4",
            "--count_lr", "1e-3", "--count_wd", "1e-4",
        ]
        if h["encoder_path"]:
            argv += ["--encoder_path", h["encoder_path"]]
        argv += self._dims_argv()
        if gene_embs_condition:
            argv += ["--gene_embs_condition", gene_embs_condition]

        print(f"[PerturbGen] extracting embeddings -> {output_dir} ...")
        with _preserve_cwd():
            from perturbgen.val import main as _emb_main
            _emb_main(argv)
        return output_dir

    # ------------------------------------------------------------------ #
    # save / load
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save the model manifest (paths, hparams, checkpoint locations)."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "perturbgen_model.json"), "w") as f:
            json.dump(
                {"tokenized_dir": self.tokenized_dir, "source": self.source,
                 "n_hvg": self.n_hvg, "hparams": self.hparams,
                 "masking_ckpt": self.masking_ckpt, "count_ckpt": self.count_ckpt},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "PerturbGen":
        with open(os.path.join(path, "perturbgen_model.json")) as f:
            m = json.load(f)
        obj = cls(m["tokenized_dir"], source=m["source"], n_hvg=m["n_hvg"], **m["hparams"])
        obj.masking_ckpt = m.get("masking_ckpt")
        obj.count_ckpt = m.get("count_ckpt")
        return obj

    # ------------------------------------------------------------------ #
    # perturbation
    # ------------------------------------------------------------------ #
    def perturb(
        self,
        genes: Sequence[str],
        *,
        mode: str = "knockout",
        where: str = "source",
        pred_tps: Optional[Sequence[int]] = None,
        pert_tps: Optional[Sequence[int]] = None,
        n_samples: int = 3,
        output_dir: str = "perturbation_out",
        count_ckpt: Optional[str] = None,
        context_mode: bool = False,
        batch_size: int = 64,
        precision: int = 16,
        temperature: float = 1.5,
        iterations: int = 19,
        extra_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Run in-silico perturbation of `genes` and return the output directory.

        - ``mode``: ``"knockout"`` (mask) | ``"overexpress"`` | ``"delete"`` | ``"pad"``.
        - ``where``: ``"source"`` (edit the source state, propagate downstream) | ``"target"``.
        - ``pert_tps``: with ``where="target"``, the predicted time points (values from
          ``pred_tps``) whose target state is perturbed, e.g. ``[1]``. Required for a
          target ``"knockout"``/``"delete"``/``"pad"``; only used with ``where="target"``.
        - ``n_samples``: number of count draws from the count decoder per cell,
          averaged into the predicted counts (1 = use the predicted mean).

        `temperature`/`iterations` have no effect here. They configure MaskGIT
        generation (``validation_mode="generate"``), but perturb() runs
        ``validation_mode="inference"``, which decodes the observed target cells,
        as in the released LPS and HSPC perturbation configs. They are kept so
        existing calls keep working.

        `context_mode` defaults to False for perturbation prediction: with True the
        decoder also attends to the observed (unperturbed) cells of the other time
        points. Training and embeddings use the model's context_mode, typically True.

        `extra_config` adds options of the perturbation config
        (``perturbgen/Perturb/val.py``) that perturb() does not set, as
        ``{section: {key: value}}`` with section ``"data"``, ``"trainer"``,
        ``"datamodule"`` or ``"model"``. Options that perturb() already sets (from
        its arguments or the ``PerturbGen(...)`` settings) are rejected. For
        example, to keep only the cell pairs whose ``drug`` is ``"Drug10"``::

            extra_config={"data": {"filter_var": "drug", "filter_cond": ["Drug10"]}}

        Other ``"data"`` keys: ``perturb_genes_file`` (CSV with the genes in its
        first column) together with ``perturb_colname`` and ``perturb_cluster``
        (integer cluster ids) perturbs the genes whose ``perturb_colname`` is in
        ``perturb_cluster``; pass ``genes=[]`` with it. ``"trainer"`` keys include
        ``exclude_src``; ``"datamodule"`` keys include ``num_workers``.

        NOTE: for a source knockout the gene must be expressed in the source
        state, otherwise every cell is filtered out. Predicted h5ads are written
        under `output_dir`.
        """
        extra_config = dict(extra_config or {})
        for section, values in extra_config.items():
            if section not in ("data", "trainer", "datamodule", "model"):
                raise ValueError(
                    f"extra_config sections must be 'data', 'trainer', 'datamodule' or "
                    f"'model', got '{section}'"
                )
            if not isinstance(values, dict):
                raise TypeError(f"extra_config['{section}'] must be a dict, got {type(values).__name__}")
        if "perturb_genes_file" in extra_config.get("data", {}):
            # Perturb/val.py changes the working directory to the repository root
            extra_config["data"] = {
                **extra_config["data"],
                "perturb_genes_file": os.path.abspath(extra_config["data"]["perturb_genes_file"]),
            }
        if (temperature, iterations) != (1.5, 19):
            warnings.warn(
                "perturb(): temperature and iterations have no effect (perturbation runs "
                "validation_mode='inference', without MaskGIT generation).",
                stacklevel=2,
            )
        ckpt = count_ckpt or self.count_ckpt
        if ckpt is None:
            raise ValueError("perturb() needs a trained count decoder; run train_count() first or pass count_ckpt=.")
        if mode not in _MODE_ALIASES:
            raise ValueError(f"mode must be one of {list(_MODE_ALIASES)}")
        if where not in _WHERE_ALIASES:
            raise ValueError(f"where must be one of {list(_WHERE_ALIASES)}")
        if _WHERE_ALIASES[where] == "tgt" and _MODE_ALIASES[mode] != "overexpress" and pert_tps is None:
            raise ValueError(
                f"where='target' with mode='{mode}' needs pert_tps=<time points to perturb>, "
                "e.g. pert_tps=[1]."
            )

        h, p = self.hparams, self.paths
        tps = list(pred_tps or h["pred_tps"])
        output_dir = os.path.abspath(output_dir)
        ckpt = os.path.abspath(ckpt)
        os.makedirs(output_dir, exist_ok=True)
        config: Dict[str, Any] = {
            "data": {
                "src_dataset_file": p["src_dataset"],
                "tgt_dataset_folder": p["tgt_dataset_folder"],
                "src_adata": p["src_adata"],
                "tgt_adata_folder": p["tgt_adata_folder"],
                **({"cond_list": h["cond_list"]} if h.get("cond_list") else {}),
            },
            "trainer": {
                "use_count_decoder": True,
                "n_samples": n_samples,
                "temperature": temperature,
                "iterations": iterations,
                "d_model": h["d_model"],
                "num_heads": h.get("num_heads", 8),   # training default (not overridden in jobscripts)
                "num_layers": h["n_layers"],
                "d_ff": h["d_ff"],
                "dropout": 0,
                "loss_mode": h["loss_mode"],
                "mask_scheduler": h["mask_scheduler"],
                "output_dir": output_dir,
                "mapping_dict_path": p["mapping_dict_path"],
                "tokenid_to_rowid_path": p["tokenid_to_rowid_path"],
                "encoder": h["encoder"],
                "encoder_path": h["encoder_path"],
                "context_mode": context_mode,
                "pos_encoding_mode": h["pos_encoding_mode"],
                "perturbation_mode": _MODE_ALIASES[mode],
                "genes_to_perturb": list(genes),
                "validation_mode": "inference",
                "perturbation_sequence": [_WHERE_ALIASES[where]],
                **({"pert_tps": [int(t) for t in pert_tps]} if pert_tps is not None else {}),
                "var_list": h["var_list"],
                "pred_tps": tps,
                # Perturb/val.py uses these base values (instead of auto-detecting) when both are set
                **({"tgt_vocab_size": h["tgt_vocab_size"], "max_seq_length": h["max_len"]}
                   if h.get("max_len") is not None else {}),
            },
            "datamodule": {
                "batch_size": batch_size, "num_workers": 4,
                "shuffle": False, "split": False,
                "var_list": h["var_list"], "pred_tps": tps,
            },
            "model": {"precision": precision, "ckpt_masking_path": ckpt},
        }
        # keys perturb() sets, plus those Perturb/val.py recomputes from the data
        owned = {s: set(v) for s, v in config.items()}
        owned["trainer"] |= {"n_genes", "max_seq_length", "tgt_vocab_size"}
        owned["datamodule"] |= {"max_len"}
        owned["datamodule"].discard("num_workers")   # a default, safe to change
        for section, values in extra_config.items():
            clash = sorted(set(values) & owned[section])
            if clash:
                raise ValueError(
                    f"extra_config['{section}'] cannot set {clash}: perturb() sets these "
                    "from its arguments or the PerturbGen(...) settings."
                )
            config[section].update(values)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(config, tf)
            cfg_path = tf.name
        print(f"[PerturbGen] perturbing {list(genes)} ({mode}, where={where}) ...")
        # Perturb/val.py reads --config from sys.argv; drive it in-process.
        old_argv = sys.argv
        try:
            sys.argv = ["perturbgen-perturb", "--config", cfg_path]
            with _preserve_cwd():
                # imported here (in-process): Perturb/val.py calls os.chdir(ROOT) at
                # import, which must not happen before output_dir is made absolute
                from perturbgen.Perturb import val as _perturb_val
                _perturb_val.main()
        finally:
            sys.argv = old_argv
            os.unlink(cfg_path)
        return output_dir


def _detect_n_hvg(tokenized_dir: str) -> int:
    """Infer n_hvg from the tokenizer's ``dataset_<n>_hvg_src`` folder name."""
    ns = set()
    for h in glob.glob(os.path.join(tokenized_dir, "dataset_*_hvg_src")):
        parts = os.path.basename(h).split("_")  # dataset_<n>_hvg_src
        if len(parts) == 4 and parts[0] == "dataset" and parts[2] == "hvg" and parts[1].isdigit():
            ns.add(int(parts[1]))
    if len(ns) == 1:
        return ns.pop()
    if not ns:
        raise FileNotFoundError(
            f"Could not auto-detect n_hvg: no 'dataset_<n>_hvg_src' folder under "
            f"{tokenized_dir}. Pass n_hvg=... explicitly."
        )
    raise ValueError(
        f"Multiple n_hvg variants under {tokenized_dir}: {sorted(ns)}. "
        f"Pass n_hvg=... explicitly."
    )


def _latest_ckpt(output_dir: str) -> Optional[str]:
    """Return the most-recent .ckpt under <output_dir>/checkpoints."""
    ckpts = glob.glob(os.path.join(output_dir, "checkpoints", "*.ckpt"))
    if not ckpts:
        ckpts = glob.glob(os.path.join(output_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)
