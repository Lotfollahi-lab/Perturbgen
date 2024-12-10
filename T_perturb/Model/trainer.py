import os
import pickle
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

import anndata as ad
import evaluate
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torchmetrics import MeanSquaredError
from torchmetrics.text import Perplexity

from T_perturb.Modules.T_model import CellGen, CountDecoder
from T_perturb.src.losses import mse_loss
from T_perturb.src.metric import (
    compute_distribution_distances,
    evaluate_emd,
    evaluate_mmd,
    lin_reg_summary,
)
from T_perturb.src.utils import (  # WarmupScheduler,;
    aggregate_attn_weights,
    compute_cos_similarity,
    modify_ckpt_state_dict,
    pearson,
    return_attn_weights,
    return_cos_similarity,
    return_gene_embeddings,
    save_cosine_similarity,
    save_gene_embeddings,
    save_generation_adata,
    save_prediction_adata,
    scale_pca,
)

# from deepspeed.ops.adam import FusedAdam


def set_matmul_precision_for_device(precision: Literal['high', 'medium'] = 'high'):
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name()
    # If the device is an A100, set the precision for matrix multiplication
    if ('A100' in cuda_device_name) or ('NVIDIA H100 80GB HBM' in cuda_device_name):
        print(f'Using {cuda_device_name} for training')
        print(f'Set float32_matmul_precision to {precision}')
        torch.set_float32_matmul_precision(precision)


class CellGenTrainer(LightningModule):
    def __init__(
        self,
        tgt_vocab_size: int = 25000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        d_ff: int = 2048,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        mlm_probability: float = 0.15,
        weight_decay: float = 0.0,
        initial_lr: float = 1e-4,
        end_lr: float = 1e-3,
        return_embeddings: bool = False,
        return_attn: bool = False,
        generate: bool = False,
        pred_tps: list = [1, 2],
        n_total_tps: int = 3,
        num_epochs: int = 5,
        warmup_epochs: int = 1,
        pad_token_id: int = 0,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        encoder: str = 'GF_fine_tuned',
        mask_scheduler: str = 'cosine',
        context_mode: bool = True,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt'
        ] = 'time_pos_sin',
        precision: Literal['high', 'medium'] = 'medium',
        var_list: Optional[List[str]] = None,
        gene_names: Optional[List[str]] = None,
        mapping_dict_path: Optional[str] = None,
        marker_genes_path: Optional[str] = None,
        context_tps: Optional[List[int]] = None,
        # *args,
        # **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        set_matmul_precision_for_device(precision)
        print('start training')
        if context_tps is None:
            context_tps = pred_tps
        self.context_tps = context_tps
        self.transformer = CellGen(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            mlm_probability=mlm_probability,
            pred_tps=pred_tps,
            context_tps=context_tps,
            n_total_tps=n_total_tps,
            encoder=encoder,
            mask_scheduler=mask_scheduler,
            pos_encoding_mode=pos_encoding_mode,
            return_attn=return_attn,
        )
        self.masking_loss = nn.CrossEntropyLoss()
        self.timepoint_loss = nn.CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.perplexity = Perplexity(ignore_index=-100)
        self.mse = MeanSquaredError()
        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                gene_to_rowid = pickle.load(f)
        with open(
            './T_perturb/T_perturb/pp/res/hspc/tokenid_to_rowid_hvg.pkl',
            'rb',
        ) as f:
            tokenid_to_rowid = pickle.load(f)
        self.tokenid_to_rowid = tokenid_to_rowid
        # swap key and value
        self.gene_to_rowid = {v: k for k, v in gene_to_rowid.items()}
        # selected marker genes for gene embeddings and attention score
        if marker_genes_path is not None:
            with open(
                marker_genes_path,
                'rb',
            ) as f:
                marker_genes = pickle.load(f)
            # concatenate all marker genes which are in a list
            marker_genes_all = [
                gene for genes in marker_genes.values() for gene in genes
            ]
            self.marker_genes_all = list(set(marker_genes_all))
        else:
            self.marker_genes_all = []

        self.return_embeddings = return_embeddings
        self.generate = generate
        self.tgt_vocab_size = tgt_vocab_size
        self.pred_tps = pred_tps
        self.n_total_tps = n_total_tps
        self.context_mode = context_mode

        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'cls_embeddings': [],
            'cosine_similarities': [],
            'batch': [],
            'cell_idx': [],
            'gene_embeddings': [],
        }

        for t in range(1, n_total_tps + 1):
            if return_attn:
                self.test_dict[f'self_attn_weights_t{t}'] = []
                self.test_dict[f'cross_attn_weights_t{t}'] = []
        if var_list is not None:
            self.var_list = var_list
            for var in self.var_list:
                self.test_dict[var] = []
        else:
            self.var_list = []

        self.pad_token_id = pad_token_id
        self.gene_names = gene_names
        total_vocab_size = tgt_vocab_size
        # register buffer for CLS
        # initialize cls token for all time steps
        self.total_tps = list(range(1, n_total_tps + 1))
        for i in self.total_tps:
            # i-1, as first token is tgt_vocab_size
            self.register_buffer(
                f'cls_token_{str(i)}',
                torch.tensor(
                    [total_vocab_size + (i - 1)],
                    dtype=torch.long,
                ),
            )
            print(f'cls_token_{str(i)}', getattr(self, f'cls_token_{str(i)}'))
            # update mapping_dict to include cls token
            self.gene_to_rowid[f'cls_token_{str(i)}'] = total_vocab_size + (i - 1)
        self.output_dir = output_dir
        # create directory if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.date = datetime.now().strftime('%Y%m%d-%H:%M')
        self.return_attn = return_attn

    def forward(self, batch):
        self.tgt_input_id_dict = {}
        for i in self.pred_tps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            self.tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=self.tgt_input_id_dict,
            not_masked=self.return_embeddings,
            context_mode=self.context_mode,
        )
        return outputs

    def configure_optimizers(self):
        parameters = [{'params': self.transformer.parameters(), 'lr': self.initial_lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        # number_of_batches_per_epoch = len(self.trainer.datamodule.train_dataloader())
        # total_steps = self.num_epochs * number_of_batches_per_epoch
        # warmup_steps = self.warmup_epochs * number_of_batches_per_epoch
        # scheduler = WarmupScheduler(
        #     optimizer,
        #     warmup_steps=warmup_steps,
        #     initial_lr=self.initial_lr,
        #     end_lr=self.end_lr,
        # )

        # optimizer = FusedAdam(
        #     self.transformer.parameters(),
        #     lr=self.initial_lr,
        #     weight_decay=self.weight_decay,
        #     adam_w_mode=False,
        # )
        return {
            'optimizer': optimizer,
            'monitor': 'train/masking_loss',
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'step',
            #     'frequency': 1
            # }
        }

    def training_step(self, batch, *args, **kwargs):
        # log learning rate
        self.log(
            'lr',
            self.trainer.optimizers[0].param_groups[0]['lr'],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        outputs = self.forward(batch)
        dec_logits = outputs['dec_logits']
        labels = outputs['labels']
        with torch.no_grad():
            perp = self.perplexity(dec_logits, labels)
            self.log(
                'train/perplexity',
                perp,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['tgt_input_ids_t1'].shape[0],
                rank_zero_only=True,
                sync_dist=True,
            )
        dec_logits = dec_logits.contiguous().view(-1, dec_logits.size(-1))
        labels = labels.contiguous().view(-1)
        masking_loss = self.masking_loss(dec_logits, labels)
        self.log(
            'train/masking_loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
            rank_zero_only=True,
            sync_dist=True,
        )
        return masking_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        dec_logits = outputs['dec_logits']
        labels = outputs['labels']
        perp = self.perplexity(dec_logits, labels)
        dec_logits = dec_logits.contiguous().view(-1, dec_logits.size(-1))
        labels = labels.contiguous().view(-1)
        masking_loss = self.masking_loss(dec_logits, labels)

        self.log(
            'val/loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
            rank_zero_only=True,
            sync_dist=True,
        )
        self.log(
            'val/perplexity',
            perp,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
            rank_zero_only=True,
            sync_dist=True,
        )
        return masking_loss

    def test_step(self, batch, *args, **kwargs):
        if self.return_embeddings:
            outputs = self.forward(batch)

            for t in self.pred_tps:
                token_ids = self.tgt_input_id_dict[f'tgt_input_ids_t{t}']
                # 1. compute cosine similarity
                cos_similarity = compute_cos_similarity(outputs=outputs, time_step=t)
                # 2. map cosine similarity to corresponding genes
                marker_cos_similarity, marker_genes_dict = return_cos_similarity(
                    cos_similarity=cos_similarity,
                    mapping_dict=self.gene_to_rowid,
                    token_ids=token_ids,
                    marker_genes=self.marker_genes_all,
                )
                if self.marker_genes_all is not None:
                    # take the non zero mean of the gene embeddings
                    gene_embeddings = return_gene_embeddings(
                        # marker_genes=marker_genes,
                        gene_embeddings=outputs['dec_embedding'][t][:, 1:, :],
                        mapping_dict=self.gene_to_rowid,
                        token_ids=token_ids,
                        marker_genes=self.marker_genes_all,
                    )
                    gene_embeddings = gene_embeddings.detach().cpu()
                    gene_embeddings = gene_embeddings.to(torch.float16)
                    self.test_dict['gene_embeddings'].append(gene_embeddings)
                if self.return_attn:
                    context_tps = [tp for tp in self.context_tps if tp != t]
                    # extract context_ids
                    context_ids = [batch['src_input_ids']]
                    context_ids.extend(
                        [
                            self.tgt_input_id_dict[f'tgt_input_ids_t{t}']
                            for t in context_tps
                        ]
                    )
                    self_attn_weights, cross_attn_weights = return_attn_weights(
                        outputs=outputs,
                        tgt_mapping_dict=self.gene_to_rowid,
                        src_mapping_dict=self.tokenid_to_rowid,
                        time_step=t,
                        token_ids=token_ids,
                        pad_token_id=self.pad_token_id,
                        context_token_ids=context_ids,
                    )
                    self_attn_weights = self_attn_weights.mean(dim=0).detach().cpu()
                    cross_attn_weights = cross_attn_weights.mean(dim=0).detach().cpu()
                    self.test_dict[f'self_attn_weights_t{t}'].append(self_attn_weights)
                    self.test_dict[f'cross_attn_weights_t{t}'].append(
                        cross_attn_weights
                    )

                cos_similarity = marker_cos_similarity.detach().cpu()
                cos_similarity = cos_similarity.to(torch.float16)

                true_counts = batch[f'tgt_counts_t{t}'].detach().cpu()
                cls_embeddings = outputs['mean_embedding'][t].detach().cpu()
                # gene_embeddings = marker_gene_embeddings.detach().cpu()
                combined_batch = batch['combined_batch'].detach().cpu()
                self.test_dict['true_counts'].append(true_counts)
                self.test_dict['cls_embeddings'].append(cls_embeddings)
                self.test_dict['cosine_similarities'].append(cos_similarity)
                # self.test_dict['gene_embeddings'].append(gene_embeddings)
                self.test_dict['batch'].append(combined_batch)
                self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{t}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{t}'])

            self.marker_genes = marker_genes_dict

    def on_test_epoch_end(self):
        if self.return_attn:
            for t in range(1, self.n_total_tps + 1):
                self_attn_weights = torch.stack(
                    self.test_dict[f'self_attn_weights_t{t}']
                )
                cross_attn_weights = torch.stack(
                    self.test_dict[f'cross_attn_weights_t{t}']
                )
                # order genes based on ascending order of  rowid
                tgt_gene_order = sorted(self.gene_to_rowid, key=self.gene_to_rowid.get)
                aggregate_attn_weights(
                    attn_weights=self_attn_weights,
                    tgt_gene_names=tgt_gene_order,
                    output_dir=self.output_dir,
                    file_name=f'{self.date}_self_attn_weights_t{t}',
                )
                aggregate_attn_weights(
                    attn_weights=cross_attn_weights,
                    tgt_gene_names=tgt_gene_order,
                    src_gene_names=tgt_gene_order[
                        : -self.n_total_tps
                    ],  # exclude cls token
                    output_dir=self.output_dir,
                    file_name=f'{self.date}_cross_attn_weights_t{t}',
                )
        if self.return_embeddings:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['batch', 'cell_idx'])
            save_dict = {
                'test_dict': self.test_dict,
                'marker_genes': self.marker_genes,
                'output_dir': self.output_dir,
                'obs_key': obs_key,
            }
            save_cosine_similarity(
                file_name=f'{self.date}_cosine_similarities',
                **save_dict,
            )
            save_prediction_adata(
                gene_names=self.gene_names,
                file_name=f'{self.date}_cell_embeddings',
                n_total_tps=self.n_total_tps,
                **save_dict,
            )
            if len(self.marker_genes_all) > 0:
                save_gene_embeddings(
                    test_dict=self.test_dict,
                    output_dir=self.output_dir,
                    file_name=f'{self.date}_gene_embeddings',
                )


class CountDecoderTrainer(LightningModule):
    def __init__(
        self,
        tgt_vocab_size: int = 25000,
        d_model=256,
        num_heads=8,
        num_layers=1,
        d_ff=32,
        max_seq_length=2048,
        loss_mode: str = 'mse',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        generate: bool = False,
        var_list: List[str] = ['Time_point'],
        pred_tps: list = [1, 2],
        n_total_tps: int = 3,
        temperature: float = 2.0,
        iterations: int = 18,
        n_samples: int = 1,
        precision: Literal['high', 'medium'] = 'medium',
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        encoder: str = 'GF_fine_tuned',
        mapping_dict_path: str = (
            './T_perturb/Geneformer/geneformer/' 'token_dictionary_gc95M.pkl'
        ),
        seed: int = 42,
        context_mode: bool = True,
        n_genes: int = 25426,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt'
        ] = 'time_pos_sin',
        mask_scheduler: Optional[str] = 'cosine',
        sequence_length: int = 2048,
        return_rouge_score=True,
        tgt_adata: Optional[ad.AnnData] = None,
        ckpt_masking_path: Optional[str] = None,
        ckpt_count_path: Optional[str] = None,
        conditions: Optional[Dict[Any, Any]] = None,
        conditions_combined: Optional[List[Any]] = None,
        unique_gene_list: Optional[Dict[Any, Any]] = None,
        shared_gene_list: Optional[Dict[Any, Any]] = None,
        context_tps: Optional[List[int]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        set_matmul_precision_for_device(precision)
        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                ensembl_to_token_id = pickle.load(f)
        # change to token_id to gene name
        self.token_id_to_ensembl = {v: k for k, v in ensembl_to_token_id.items()}
        pretrained_model = CellGen(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            pred_tps=pred_tps,
            context_tps=context_tps,
            n_total_tps=n_total_tps,
            encoder=encoder,
            pos_encoding_mode=pos_encoding_mode,
        )
        self.pos_encoding_mode = pos_encoding_mode
        # load PETRA checkpoint
        if ckpt_masking_path is not None:
            checkpoint = torch.load(ckpt_masking_path, map_location='cpu')
            state_dict_ = modify_ckpt_state_dict(checkpoint, 'transformer.')
            pretrained_model.load_state_dict(state_dict_, strict=False)
            # set parameters to not trainable
            for param in pretrained_model.parameters():
                param.requires_grad = False
        self.return_rouge_score = return_rouge_score
        if self.return_rouge_score:
            self.rouge = evaluate.load('rouge')
        self.decoder = CountDecoder(
            pretrained_model=pretrained_model,
            loss_mode=loss_mode,
            d_model=d_model,
            dropout=dropout,
            pred_tps=pred_tps,
            context_tps=context_tps,
            context_mode=context_mode,
            n_genes=n_genes,
        )
        if ckpt_count_path is not None:
            checkpoint = torch.load(ckpt_count_path, map_location='cpu')

            state_dict_ = modify_ckpt_state_dict(checkpoint, 'decoder.')
            self.decoder.load_state_dict(state_dict_, strict=False)

        self.weight_decay = weight_decay
        self.lr = lr
        self.loss_mode = loss_mode
        self.max_seq_length = max_seq_length
        if (
            (self.loss_mode in ['nb', 'zinb'])
            and (conditions is not None)
            and (conditions_combined is not None)
        ):
            self.n_conditions = [len(conditions[cond]) for cond in conditions.keys()]
            self.n_conditions_combined = len(conditions_combined)

            self.theta = torch.nn.Parameter(
                torch.randn(n_genes, self.n_conditions_combined)
            )
        else:
            self.theta = None

        self.mse = MeanSquaredError()
        total_vocab_size = tgt_vocab_size
        self.pred_tps = pred_tps
        self.total_tps = list(range(1, n_total_tps + 1))
        for i in self.total_tps:
            self.register_buffer(
                f'cls_token_{str(i)}',
                torch.tensor(
                    [total_vocab_size + (i - 1)],
                    dtype=torch.long,
                ),
            )
        self.generate = generate
        self.adata = tgt_adata
        # scheduler
        self.mask_scheduler = mask_scheduler
        self.temperature = temperature
        self.iterations = iterations
        self.sequence_length = sequence_length
        self.output_dir = output_dir
        # create directory if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # create variables based
        # initiate lists to store true, ctrl and pred counts
        self.train_dict: Dict[str, List[Any]] = {'true_counts': [], 'pred_counts': []}
        self.val_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'pred_counts': [],
        }
        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'ctrl_counts': [],
            'pred_counts': [],
            'cls_embeddings': [],
            'cell_idx': [],
        }
        if self.return_rouge_score:
            self.rouge_seq_len_list = [25, 100, max_seq_length]
            for seq_len in self.rouge_seq_len_list:
                self.test_dict[f'rouge1_{seq_len}'] = []
        self.n_samples = n_samples
        if var_list is not None:
            self.var_list = var_list
            for var in self.var_list:
                self.test_dict[var] = []
        else:
            self.var_list = []
        self.encoder = encoder
        self.seed = seed
        self.date = datetime.now().strftime('%Y%m%d-%H:%M')
        # guided generation

        self.unique_gene_list = unique_gene_list
        self.shared_gene_list = shared_gene_list

    def forward(self, batch):
        tgt_input_id_dict = {}
        for i in self.pred_tps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        outputs = self.decoder(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
        )

        return outputs

    def one_hot_encoder(
        self,
        idx,
        n_cls,
        dtype,
    ):
        assert torch.max(idx) < n_cls

        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        self.register_buffer(
            'onehot', torch.zeros(idx.size(0), n_cls, dtype=dtype, device=idx.device)
        )
        # change idx dtype to onehot dtype
        idx = idx.type(self.onehot.dtype)
        self.onehot.scatter_(1, idx.long(), 1)
        return self.onehot

    def compute_dispersion(self, batch):
        dispersions = F.linear(
            self.one_hot_encoder(
                batch['combined_batch'],
                self.n_conditions_combined,
                self.theta.dtype,
            ),
            self.theta,
        )
        return torch.exp(dispersions)

    def compute_count_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        n_samples: int = 1,
    ):
        """
        Description:
        ------------
        Use CLS or mean non padding embeddings to predict gene counts.

        Parameters:
        -----------
        outputs: `Dict[str, torch.Tensor]`
            model outputs
        batch: `Dict[str, torch.Tensor]`
            batch variables capturing technical batch effect variables
        n_samples: `int`
            number of samples to draw from distribution for zinb and nb loss

        Returns:
        --------
        `Tuple[torch.Tensor, Dict[str, torch.Tensor]]` \n
            - loss: `torch.Tensor`
                loss value
            - count_dict: `Dict[str, torch.Tensor]`
                dictionary containing predicted counts
        """
        count_dict = {}
        dispersion = (
            self.compute_dispersion(batch) if self.loss_mode in ['zinb', 'nb'] else None
        )
        total_loss = 0
        for time_step in self.pred_tps:
            count_ouput = outputs[f'count_output_t{time_step}']
            true_values = batch[f'tgt_counts_t{time_step}']
            batch_size_factor = batch[f'tgt_size_factor_t{time_step}'].unsqueeze(1)

            if self.loss_mode == 'mse':
                # change true counts dtype to count output dtype
                true_values = true_values.type(count_ouput['count_lognorm'].dtype)
                loss = (
                    mse_loss(count_ouput['count_lognorm'], true_values)
                    .sum(dim=-1)
                    .mean()
                    .float()
                )
                count_dict[time_step] = count_ouput['count_lognorm']
            elif self.loss_mode in ['zinb', 'nb']:
                dec_mean = count_ouput['count_mean'] * batch_size_factor.expand_as(
                    count_ouput['count_mean']
                )
                if self.loss_mode == 'zinb':
                    dec_dropout = count_ouput['count_dropout']
                    zinb_distribution = ZeroInflatedNegativeBinomial(
                        mu=dec_mean,
                        theta=dispersion,
                        zi_logits=dec_dropout,
                    )
                    loss = -zinb_distribution.log_prob(true_values).sum(dim=-1).mean()
                    if n_samples == 1:
                        count_dict[time_step] = dec_mean
                    else:
                        # sample from distribution
                        x_pred = zinb_distribution.sample((n_samples,))
                        count_dict[time_step] = x_pred.mean(dim=0)

                elif self.loss_mode == 'nb':
                    nb_distribution = NegativeBinomial(mu=dec_mean, theta=dispersion)
                    loss = -nb_distribution.log_prob(true_values).sum(dim=-1).mean()
                    if n_samples == 1:
                        count_dict[time_step] = dec_mean
                    else:
                        x_pred = nb_distribution.sample((n_samples,))
                        count_dict[time_step] = x_pred.mean(dim=0)
            total_loss += loss
        return total_loss, count_dict

    def compute_mse_metric(
        self,
        pred_counts: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        true_counts_key: str,
        time_steps: list[str],
        res_dict: dict[str, list],
        save_to_cpu: bool = False,
    ) -> tuple[float, dict[str, list[Any]]]:
        total_mse = 0.0
        for time_step in time_steps:
            pred_count = pred_counts[time_step]
            true_count = batch[f'{true_counts_key}_t{time_step}']
            # MSE
            mse = self.mse(pred_count, true_count)
            total_mse += mse
            if save_to_cpu:
                pred_count = pred_count.detach().cpu()
                true_count = true_count.detach().cpu()
            res_dict['pred_counts'].append(pred_count)
            res_dict['true_counts'].append(true_count)
        mean_mse = total_mse / len(time_steps)
        return mean_mse, res_dict

    def map_token_to_ensembl(self, val):
        return self.token_id_to_ensembl.get(
            val, val
        )  # Return mapped value, or original if not in dict

    def compute_rouge_score(
        self,
        pred_ids: np.ndarray,
        tgt_ids: np.ndarray,
        rouge_len_list: list[int],
        max_seq_length: int,
        test_dict: dict[str, list],
    ) -> tuple[dict[str, list[Any]], dict[Any, Any]]:
        rouge_score = {}
        pred_ids = pred_ids.astype(object)
        pred_ids = pred_ids[:, 1:]  # exclude task token
        tgt_ids = tgt_ids.astype(object)
        special_tokens = np.array([0, 1, 2, 3])
        pred_ids[np.isin(pred_ids, special_tokens)] = ''
        tgt_ids[np.isin(tgt_ids, special_tokens)] = ''
        # convert all int to str
        pred_ids_ = pred_ids.astype(str)
        tgt_ids_ = tgt_ids.astype(str)
        # # Vectorize the function to apply to the entire matrix
        # vectorized_map = np.vectorize(self.map_token_to_ensembl)
        # # TODO: rewrite the function without mapping dict
        # # Apply the mapping
        # pred_ids_ = vectorized_map(pred_ids)
        # tgt_ids_ = vectorized_map(tgt_ids)
        for seq_len in rouge_len_list:
            if max_seq_length > seq_len:
                pred_genes_short = pred_ids_[:, :seq_len]
                true_genes_short = tgt_ids_[:, :seq_len]
            else:
                pred_genes_short = pred_ids_
                true_genes_short = tgt_ids_
            pred_ids_str = np.apply_along_axis(
                lambda row: ' '.join(row), axis=1, arr=pred_genes_short
            )
            tgt_ids_str = np.apply_along_axis(
                lambda row: ' '.join(row), axis=1, arr=true_genes_short
            )
            # remove all the trailing spaces
            pred_ids_str = np.array([' '.join(s.split()) for s in pred_ids_str])
            tgt_ids_str = np.array([' '.join(s.split()) for s in tgt_ids_str])
            # create a list of strings
            pred_ids_str = pred_ids_str.tolist()
            tgt_ids_str = tgt_ids_str.tolist()
            # compute rouge score
            rouge_score = self.rouge.compute(
                predictions=pred_ids_str,
                references=tgt_ids_str,
                rouge_types=['rouge1'],
            )
            test_dict[f'rouge1_{seq_len}'].append(rouge_score['rouge1'])
        return test_dict, rouge_score

    def training_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        count_loss, pred_counts_dict = self.compute_count_loss(outputs, batch)
        self.log(
            'train/loss',
            count_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
            sync_dist=True,
        )
        with torch.no_grad():
            mean_mse, res_dict = self.compute_mse_metric(
                pred_counts_dict, batch, 'tgt_counts', self.pred_tps, self.train_dict
            )
            self.train_dict = res_dict
            self.log(
                'train/mse',
                mean_mse,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return count_loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            # return Pearson correlation coefficient
            true_counts = torch.cat(self.train_dict['true_counts'])
            pred_counts = torch.cat(self.train_dict['pred_counts'])
            # Pearson correlation coefficient
            mean_pearson = pearson(pred_counts=pred_counts, true_counts=true_counts)
            self.log(
                'train/pearson',
                mean_pearson,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            # set to status quo
            self.train_true_counts_list = []
            self.train_pred_counts_list = []

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        count_loss, pred_counts_dict = self.compute_count_loss(
            outputs,
            batch,
            n_samples=self.n_samples,
        )
        self.log(
            'val/loss',
            count_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
            sync_dist=True,
        )
        mean_mse, res_dict = self.compute_mse_metric(
            pred_counts_dict,
            batch,
            'tgt_counts',
            self.pred_tps,
            self.val_dict,
        )
        self.val_dict = res_dict
        self.log(
            'val/mse',
            mean_mse,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.val_true_counts_list)
        pred_counts = torch.cat(self.val_pred_counts_list)
        mean_pearson = pearson(pred_counts=pred_counts, true_counts=true_counts)
        self.log(
            'val/pearson',
            mean_pearson,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.val_true_counts_list = []
        self.val_pred_counts_list = []

    def test_step(self, batch, *args, **kwargs):
        tgt_input_id_dict = {}
        for i in self.total_tps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if self.generate:
            decoder_kwargs = {
                'src_input_id': batch['src_input_ids'],
                'tgt_input_id_dict': tgt_input_id_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
                'sequence_length': self.sequence_length,
            }
            if (self.unique_gene_list is not None) or (
                self.shared_gene_list is not None
            ):
                decoder_kwargs['unique_gene_list'] = self.unique_gene_list
                decoder_kwargs['shared_gene_list'] = self.shared_gene_list
                outputs, pred_ids_dict = self.decoder.guided_generate(
                    **decoder_kwargs,
                )
            else:
                outputs, pred_ids_dict = self.decoder.generate(
                    **decoder_kwargs,
                )
            # print(pred_ids_dict)
            for time_step in pred_ids_dict.keys():
                if self.return_rouge_score:
                    pred_ids = pred_ids_dict[time_step].detach().cpu().numpy()
                    tgt_ids = batch[time_step].detach().cpu().numpy()
                    test_dict, rouge_score = self.compute_rouge_score(
                        pred_ids=pred_ids,
                        tgt_ids=tgt_ids,
                        rouge_len_list=self.rouge_seq_len_list,
                        max_seq_length=self.max_seq_length,
                        test_dict=self.test_dict,
                    )
                    self.test_dict = test_dict
                    self.log(
                        'test/rouge1',
                        rouge_score['rouge1'],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        rank_zero_only=True,
                        sync_dist=True,
                        batch_size=batch['src_input_ids'].shape[0],
                    )
            count_loss, pred_counts_dict = self.compute_count_loss(
                outputs=outputs,
                batch=batch,
                n_samples=self.n_samples,
            )
            self.log(
                'test/loss',
                count_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch[f'tgt_input_ids_t{self.pred_tps[0]}'].shape[0],
            )
            mean_mse, res_dict = self.compute_mse_metric(
                pred_counts_dict,
                batch,
                'tgt_counts',
                self.pred_tps,
                self.test_dict,
                save_to_cpu=True,
            )
            self.test_dict = res_dict
            self.log(
                'test/mse',
                mean_mse,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            for time_step in self.pred_tps:
                self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{time_step}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])
                cls_embeddings = outputs[f'cls_embedding_t{time_step}'].detach().cpu()
                self.test_dict['cls_embeddings'].append(cls_embeddings)
        else:
            outputs = self.forward(batch)
            count_loss, pred_count = self.compute_count_loss(
                outputs,
                batch,
                n_samples=self.n_samples,
            )
            self.log(
                'test/loss',
                count_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['src_input_ids'].shape[0],
            )
            for time_step in self.pred_tps:
                self.test_dict['pred_counts'].append(pred_count[time_step])
                self.test_dict['true_counts'].append(batch[f'tgt_counts_t{time_step}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])

    def on_test_epoch_end(self):
        if self.generate:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['cell_idx'])
            pred_adata = save_generation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_generate_adata_'
                    f't{self.pred_tps}_{self.encoder}_s{self.seed}_'
                    f'l{self.loss_mode}_n{self.n_samples}'
                    f'_p{self.pos_encoding_mode}_'
                    f'm{self.mask_scheduler}_s{self.sequence_length}'
                ),
            )
            # save metrics
            metric_mean = {}
            # true counts are stored in the 'counts' layer
            true_adata = pred_adata.copy()
            true_adata.X = true_adata.layers['counts']
            # log norm and compute PCA
            pred_adata = scale_pca(pred_adata)
            true_adata = scale_pca(true_adata)
            # scale pca
            coords = true_adata.obsm['X_pca']
            coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
            true_adata.obsm['X_pca_scaled'] = coords
            coords = pred_adata.obsm['X_pca']
            coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
            pred_adata.obsm['X_pca_scaled'] = coords

            # subsample 25k cells
            if pred_adata.shape[0] > 10000:
                sc.pp.subsample(pred_adata, n_obs=10000, copy=False)
                # use obs index to subsample true counts
                true_adata = true_adata[pred_adata.obs.index]
            mmd_wasserstein = compute_distribution_distances(
                torch.tensor(true_adata.obsm['X_pca_scaled']).float(),
                torch.tensor(pred_adata.obsm['X_pca_scaled']).float(),
            )
            for metric in mmd_wasserstein:
                metric_mean[metric + '_PCA'] = mmd_wasserstein[metric]
            emd_df = evaluate_emd(true_adata, pred_adata)
            self.log(
                'test/emd',
                emd_df['emd'].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            print('---Metrics saved')
            if self.return_rouge_score:
                if self.test_dict[f'rouge1_{self.rouge_seq_len_list[0]}']:
                    for seq_len in self.rouge_seq_len_list:
                        metric_mean[f'rouge1_{seq_len}'] = np.mean(
                            self.test_dict[f'rouge1_{seq_len}']
                        )

            metrics = pd.DataFrame(metric_mean, index=[0])
            # add metrics on the gene space
            lin_reg_df = lin_reg_summary(true_adata, pred_adata)
            mmd_df = evaluate_mmd(true_adata, pred_adata, n_cells=10000)
            metrics = pd.concat([metrics, emd_df, lin_reg_df, mmd_df], axis=1)
            metrics.to_csv(
                f'{self.output_dir}/{self.date}_p{self.pos_encoding_mode}_'
                f'm{self.mask_scheduler}_t{self.temperature}_i{self.iterations}'
                f'_s{self.seed}_s{self.sequence_length}_metrics.csv'
            )

        # else:
        #     var_dict = {}
        #     for var in self.var_list:
        #         var_dict[var] = np.concatenate(self.test_dict[var])
        #     test_obs = pd.DataFrame(var_dict)
        #     pred_adata = ad.AnnData(X=pred_counts.numpy(), obs=test_obs)
        #     pred_adata.layers['counts'] = true_counts.numpy()
        #     pred_adata.write_h5ad(f'{self.output_dir}/pred_adata.h5ad')
        #     # true counts are stored in the 'counts' layer
        #     true_adata = pred_adata.copy()
        #     true_adata.X = true_adata.layers['counts']
        #     # ----------------- calculate metrics -----------------
        #     # MSE
        #     lin_reg_df = lin_reg_summary(true_adata, pred_adata)
        #     mmd_df = evaluate_mmd(true_adata, pred_adata, n_cells=10000)
        #     emd = evaluate_emd(true_adata, pred_adata)
        #     metric_df = pd.concat([lin_reg_df, mmd_df, emd], axis=1)
        #     metric_df.to_csv(f'{self.output_dir}/test_metrics.csv')
        #     emd['metric'] = 'emd'
        #     emd = emd.rename(columns={'emd': 'value'})
        #     self.log(
        #         'test/emd',
        #         emd['value'].mean(),
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True,
        #     )

    def configure_optimizers(self):
        # optimizer = FusedAdam(
        #     self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        parameters = [{'params': self.decoder.parameters(), 'lr': self.lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        return {
            'optimizer': optimizer,
            'monitor': 'train/loss',
        }
