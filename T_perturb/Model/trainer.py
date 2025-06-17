import os
import pickle
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
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

from T_perturb.Modules.T_model import CountDecoder, CytoMeister
from T_perturb.src.losses import mse_loss
from T_perturb.src.metric import (
    compute_distribution_distances,
    compute_emd,
    evaluate_emd,
    evaluate_mmd,
    lin_reg_summary,
)
from T_perturb.src.utils import (  # WarmupScheduler
    aggregate_attn_weights,
    compute_rouge_score,
    concat_cond_tokens,
    exclude_special_tokens,
    modify_ckpt_state_dict,
    return_attn_weights,
    return_gene_embeddings,
    return_generation_adata,
    return_prediction_adata,
    scale_pca,
)

# from deepspeed.ops.adam import FusedAdam


def set_matmul_precision_for_device(precision: Literal['high', 'medium'] = 'medium'):
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name()
        if ('A100' in cuda_device_name) or ('NVIDIA H100 80GB HBM' in cuda_device_name):
            print(f'Using {cuda_device_name} for training')
            print(f'Set float32_matmul_precision to {precision}')
            torch.set_float32_matmul_precision(precision)

    else:
        print('CUDA is not available, using CPU for training.')


class CytoMeisterTrainer(LightningModule):
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
        return_gene_embs: bool = False,
        return_attn: bool = False,
        generate: bool = False,
        pred_tps: list = [1, 2],
        n_total_tps: int = 3,
        num_epochs: int = 5,
        warmup_epochs: int = 1,
        pad_token_id: int = 0,
        temperature: float = 2.0,
        iterations: int = 18,
        sequence_length: int = 2048,
        return_rouge_score: bool = True,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        encoder: Literal['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'] = (
            'GF_fine_tuned'
        ),
        mask_scheduler: str = 'cosine',
        context_mode: bool = True,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt'
        ] = 'time_pos_sin',
        precision: Literal['high', 'medium'] = 'medium',
        tokenid_to_rowid_path: str = (
            'T_perturb/T_perturb/pp/res/hspc/tokenid_to_rowid_hvg.pkl'
        ),
        encoder_path: str | None = None,
        deg_pkl_path: str | None = None,
        var_list: List[str] | None = None,
        gene_names: List[str] | None = None,
        mapping_dict_path: str | None = None,
        context_tps: List[int] | None = None,
        condition_dict: Dict[str, Dict] | None = None,
        gene_embs_list: List[str] | None = None,
        gene_embs_condition: str | None = None,
        seed: int = 42,
        # *args,
        # **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        set_matmul_precision_for_device(precision)
        if context_tps is None:
            context_tps = pred_tps
            self.total_tps = pred_tps
        else:
            self.total_tps = context_tps + pred_tps
        self.pred_tps = pred_tps
        self.n_total_tps = n_total_tps
        self.context_tps = context_tps
        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                gene_to_rowid = pickle.load(f)
                # swap key and value
                self.gene_to_rowid: Dict[Any, Any] | None = {
                    v: k for k, v in gene_to_rowid.items()
                }
        else:
            self.gene_to_rowid = None
        self.transformer = CytoMeister(
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
            encoder_path=encoder_path,
            mask_scheduler=mask_scheduler,
            pos_encoding_mode=pos_encoding_mode,
            return_attn=return_attn,
            context_mode=context_mode,
            condition_dict=condition_dict,
            gene_to_rowid=self.gene_to_rowid,
            seed=seed,
        )
        self.masking_loss = nn.CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.perplexity = Perplexity(ignore_index=-100)
        self.mse = MeanSquaredError()

        with open(
            tokenid_to_rowid_path,
            'rb',
        ) as f:
            tokenid_to_rowid = pickle.load(f)
        self.tokenid_to_rowid = tokenid_to_rowid
        self.return_embeddings = return_embeddings
        self.return_gene_embs = return_gene_embs
        self.gene_embs_list = gene_embs_list
        self.gene_embs_condition = gene_embs_condition
        self.d_model = d_model
        self.generate = generate
        self.tgt_vocab_size = tgt_vocab_size

        self.context_mode = context_mode

        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'cls_embeddings': [],
            'cosine_similarities': [],
            'batch': [],
            'cell_idx': [],
        }

        self.test_dict['gene_embeddings'] = []

        self.test_dict['self_attn_weights'] = []
        self.test_dict['cross_attn_weights'] = []
        if var_list is not None:
            self.var_list = var_list
            for var in self.var_list:
                self.test_dict[var] = []
        else:
            self.var_list = []

        self.pad_token_id = pad_token_id
        self.gene_names = gene_names
        if deg_pkl_path is not None:
            # load marker genes
            with open(
                deg_pkl_path,
                'rb',
            ) as f:
                marker_genes_dict = pickle.load(f)
            marker_genes_all = [
                gene for genes in marker_genes_dict.values() for gene in genes
            ]
            marker_genes_all = list(set(marker_genes_all))
            self.marker_genes: List[str] | None = marker_genes_all
        else:
            self.marker_genes = None
        self.output_dir = output_dir
        # create directory if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.date = datetime.now().strftime('%Y%m%d-%H:%M')
        self.return_attn = return_attn
        self.mask_scheduler = mask_scheduler
        self.encoder = encoder
        self.condition_dict = condition_dict

        # generation parameters
        self.return_rouge_score = return_rouge_score
        self.temperature = temperature
        self.iterations = iterations
        self.sequence_length = sequence_length
        self.max_seq_length = max_seq_length
        if self.return_rouge_score:
            # load rouge
            self.rouge = evaluate.load('rouge')
            # initiate rouge dict
            self.rouge_seq_len_list = [25, 100, max_seq_length]
            for seq_len in self.rouge_seq_len_list:
                self.test_dict[f'rouge1_{seq_len}'] = []

        if self.return_gene_embs:
            if self.gene_embs_list is not None:
                if self.gene_to_rowid is not None:
                    genes_dict = exclude_special_tokens(
                        self.gene_to_rowid,
                        self.marker_genes,
                    )
                    n_genes = len(genes_dict)
                    self.sum_gene_embs: Dict[str, torch.Tensor] = {
                        f'{condition}': torch.zeros(
                            size=(n_genes, self.d_model), dtype=self.dtype
                        )
                        for condition in self.gene_embs_list
                    }
                    self.count_gene_embs = {
                        f'{condition}': torch.zeros(size=(n_genes, 1), dtype=self.dtype)
                        for condition in self.gene_embs_list
                    }
                else:
                    raise ValueError('gene_to_rowid is None')
            else:
                raise ValueError('gene_embs_list is None')

    def forward(self, batch, generate: bool = False):
        tgt_input_id_dict = {}
        for i in self.total_tps:
            tgt_input_id_ = batch[f'tgt_input_ids_t{i}'].clone()
            if self.condition_dict is not None:
                cond_ids = concat_cond_tokens(
                    batch=batch,
                    time_step=i,
                    condition_dict=self.condition_dict,
                )
                tgt_input_id_ = torch.cat((cond_ids, tgt_input_id_), dim=1)
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_

        if generate:
            outputs = None
        else:
            outputs = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=self.return_embeddings,
            )
        return outputs, tgt_input_id_dict

    def configure_optimizers(self):
        parameters = [{'params': self.transformer.parameters(), 'lr': self.initial_lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)

        return {
            'optimizer': optimizer,
            'monitor': 'train/masking_loss',
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

        outputs, _ = self.forward(batch)
        for t in outputs.keys():
            dec_logits = outputs[t]['dec_logits']
            labels = outputs[t]['labels']
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
        outputs, _ = self.forward(batch)
        for t in outputs.keys():
            dec_logits = outputs[t]['dec_logits']
            labels = outputs[t]['labels']
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
        outputs, tgt_input_id_dict = self.forward(
            batch,
            generate=self.generate,
        )
        if self.condition_dict is not None:
            cond_length = len(self.condition_dict)

        else:
            cond_length = 0
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

            outputs, tgt_input_id_dict = self.transformer.generate(
                cond_length=cond_length,
                **decoder_kwargs,
            )

        for t in self.pred_tps:
            token_ids = tgt_input_id_dict[f'tgt_input_ids_t{t}']

            if self.return_gene_embs:
                # take the non zero mean of the gene embeddings
                gene_embeddings, marker_genes_dict = return_gene_embeddings(
                    gene_embeddings=outputs[t]['dec_embedding'][:, cond_length:, :],
                    mapping_dict=(
                        self.gene_to_rowid if self.gene_to_rowid is not None else None
                    ),
                    token_ids=token_ids,
                    marker_genes=self.marker_genes,
                )
                self.marker_genes_dict = marker_genes_dict
            if self.return_attn:
                context_tps = [tp for tp in self.context_tps if tp != t]
                # extract context_ids
                context_ids = [batch['src_input_ids']]
                context_ids.extend(
                    [tgt_input_id_dict[f'tgt_input_ids_t{t}'] for t in context_tps]
                )
                self_attn_weights, cross_attn_weights = return_attn_weights(
                    outputs=outputs,
                    tgt_mapping_dict=(
                        self.gene_to_rowid if self.gene_to_rowid is not None else None
                    ),
                    src_mapping_dict=self.tokenid_to_rowid,
                    time_step=t,
                    token_ids=token_ids,
                    pad_token_id=self.pad_token_id,
                    context_token_ids=context_ids,
                )
                self_attn_weights = self_attn_weights.mean(dim=0).detach().cpu()
                cross_attn_weights = cross_attn_weights.mean(dim=0).detach().cpu()
                self.test_dict['self_attn_weights'].append(self_attn_weights)
                self.test_dict['cross_attn_weights'].append(cross_attn_weights)
            # cell_idx = np.array(batch[f'tgt_cell_idx_t{t}'])
            # if len(self.test_dict['cell_idx']) == 0:
            #     all_cell_idx = np.array([])
            # else:
            #     all_cell_idx = np.concatenate(self.test_dict['cell_idx'])
            # (dupl_outside_batch, cell_idx_filter_) = mask_duplicates_across_batches(
            #     all_cell_idx, cell_idx
            # )
            # (dupl_within_batch, cell_idx_filter_) = mask_duplicates_within_batches(
            #     cell_idx_filter_
            # )
            # if self.return_embeddings:
            #     # 1. compute cosine similarity
            #     # TODO: compute cosine similarity
            #     cos_similarity = compute_cos_similarity(outputs=outputs, time_step=t)
            #     # 2. map cosine similarity to corresponding genes
            #     marker_cos_similarity, marker_genes_dict = map_results_to_genes(
            #         res=cos_similarity,
            #         mapping_dict=(
            #             self.gene_to_rowid if self.gene_to_rowid is not None else None
            #         ),
            #         token_ids=token_ids,
            #         marker_genes=self.marker_genes,
            #     )
            #     cos_similarity = marker_cos_similarity.detach().cpu()
            #     # remove duplicates
            #     cos_similarity = cos_similarity[dupl_outside_batch]
            #     cos_similarity = cos_similarity[dupl_within_batch]
            #     self.test_dict['cosine_similarities'].append(cos_similarity)
            #     self.marker_genes_dict = marker_genes_dict

            if self.return_gene_embs:
                gene_embeddings = gene_embeddings.detach().cpu()
                condition_array = np.array(batch[f'{self.gene_embs_condition}_t{t}'])
                # remove duplicates
                # gene_embeddings = gene_embeddings[dupl_outside_batch]
                # condition_array = condition_array[dupl_outside_batch]
                # gene_embeddings = gene_embeddings[dupl_within_batch]
                # condition_array = condition_array[dupl_within_batch]
                for condition in self.gene_embs_list:
                    condition_mask = condition_array == condition
                    if any(condition_mask):
                        self.sum_gene_embs[condition] += gene_embeddings[
                            condition_mask
                        ].sum(dim=0)
                        non_zero_ids = torch.nonzero(
                            gene_embeddings[condition_mask].sum(dim=2)
                        )
                        non_zero_counts = torch.bincount(
                            non_zero_ids[:, 1], minlength=gene_embeddings.shape[1]
                        )
                        self.count_gene_embs[condition] += non_zero_counts.unsqueeze(1)
            if self.generate:
                if self.return_rouge_score:
                    pred_ids = (
                        tgt_input_id_dict[f'tgt_input_ids_t{t}'].detach().cpu().numpy()
                    )
                    tgt_ids = batch[f'tgt_input_ids_t{t}'].detach().cpu().numpy()
                    # # TODO: take mean of duplicates
                    # pred_ids = pred_ids[dupl_outside_batch]
                    # tgt_ids = tgt_ids[dupl_outside_batch]
                    # pred_ids = pred_ids[dupl_within_batch]
                    # tgt_ids = tgt_ids[dupl_within_batch]
                    test_dict = compute_rouge_score(
                        rouge=self.rouge,
                        pred_ids=pred_ids,
                        tgt_ids=tgt_ids,
                        rouge_len_list=self.rouge_seq_len_list,
                        max_seq_length=self.max_seq_length,
                        test_dict=self.test_dict,
                    )
                    self.test_dict = test_dict
            true_counts = batch[f'tgt_counts_t{t}'].detach().cpu()
            cls_embeddings = outputs[t]['mean_embedding'].detach().cpu()
            combined_batch = batch['combined_batch'].detach().cpu()
            # # remove duplicates
            # true_counts = true_counts[dupl_outside_batch]
            # cls_embeddings = cls_embeddings[dupl_outside_batch]
            # combined_batch = combined_batch[dupl_outside_batch]
            # true_counts = true_counts[dupl_within_batch]
            # cls_embeddings = cls_embeddings[dupl_within_batch]
            # combined_batch = combined_batch[dupl_within_batch]

            self.test_dict['true_counts'].append(true_counts)
            self.test_dict['cls_embeddings'].append(cls_embeddings)
            self.test_dict['batch'].append(combined_batch)
            self.test_dict['cell_idx'].append(np.array(batch[f'tgt_cell_idx_t{t}']))
            if len(self.var_list) > 0:
                for var in self.var_list:
                    var_values = np.array(batch[f'{var}_t{t}'])
                    # # remove duplicates
                    # var_values = var_values[dupl_outside_batch]
                    # var_values = var_values[dupl_within_batch]
                    self.test_dict[var].append(var_values)

    def on_test_epoch_end(self):
        if self.return_attn:
            self_attn_weights = torch.stack(self.test_dict['self_attn_weights'])
            cross_attn_weights = torch.stack(self.test_dict['cross_attn_weights'])
            if self.gene_to_rowid is not None:
                # order genes based on ascending order of  rowid
                tgt_gene_order = sorted(self.gene_to_rowid, key=self.gene_to_rowid.get)
            else:
                tgt_gene_order = self.gene_names
            aggregate_attn_weights(
                attn_weights=self_attn_weights,
                tgt_gene_names=tgt_gene_order,
                output_dir=self.output_dir,
                file_name=f'{self.date}_self_attn_weights',
            )
            aggregate_attn_weights(
                attn_weights=cross_attn_weights,
                tgt_gene_names=tgt_gene_order,
                src_gene_names=tgt_gene_order[: -self.n_total_tps],  # exclude cls token
                output_dir=self.output_dir,
                file_name=f'{self.date}_cross_attn_weights',
            )
        obs_key = self.var_list if len(self.var_list) > 0 else []
        obs_key.extend(['batch', 'cell_idx'])
        if self.return_embeddings:
            # create folder to save gene embeddings
            condition_dir = f'{self.output_dir}/conditions'
            os.makedirs(condition_dir, exist_ok=True)
            # compute mean gene embeddings
            for condition in self.gene_embs_list:
                self.sum_gene_embs[condition] = np.where(
                    self.count_gene_embs[condition] != 0,
                    self.sum_gene_embs[condition] / self.count_gene_embs[condition],
                    0,
                )

            return_prediction_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                marker_genes=self.marker_genes_dict,
                # gene_names=self.gene_names,
                output_dir=self.output_dir,
                sum_gene_embs=self.sum_gene_embs,
                file_name=f'{self.date}_inference_embs_'
                f't{self.pred_tps}_{self.encoder}_'
                f'm{self.mask_scheduler}',
            )
        if self.generate:
            return_generation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_generate_adata_'
                    f't{self.pred_tps}_{self.encoder}_'
                    f'm{self.mask_scheduler}_s{self.sequence_length}'
                    f't{self.temperature}_i{self.iterations}'
                ),
            )
            # save metrics
            rouge_dict = {}
            if self.return_rouge_score:
                if self.test_dict[f'rouge1_{self.rouge_seq_len_list[0]}']:
                    for seq_len in self.rouge_seq_len_list:
                        rouge_score = np.concatenate(
                            self.test_dict[f'rouge1_{seq_len}']
                        )
                        rouge_dict[f'rouge1_{seq_len}'] = np.mean(rouge_score, axis=0)
                metrics = pd.DataFrame(rouge_dict, index=[0])
                metrics.to_csv(
                    f'{self.output_dir}/{self.date}_'
                    f'm{self.mask_scheduler}_t{self.temperature}_i{self.iterations}'
                    f'_s{self.sequence_length}_metrics.csv'
                )
                print('---Rouge score saved')


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
        d_condc: int | None = None,
        d_condt: int = 768,
        use_positional_encoding: bool = False,
        layer_norm: bool = False,
        add_cell_time: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        generate: bool = False,
        pred_tps: list = [1, 2],
        n_total_tps: int = 3,
        temperature: float = 2.0,
        iterations: int = 18,
        n_samples: int = 1,
        precision: Literal['high', 'medium'] = 'medium',
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        encoder: Literal['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'] = (
            'GF_fine_tuned'
        ),
        seed: int = 42,
        n_genes: int = 25426,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt'
        ] = 'time_pos_sin',
        mask_scheduler: str = 'cosine',
        sequence_length: int = 2048,
        return_rouge_score: bool = True,
        context_mode: bool = True,
        encoder_path: str | None = None,
        var_list: List[str] | None = None,
        tgt_adata: ad.AnnData | None = None,
        ckpt_masking_path: str | None = None,
        ckpt_count_path: str | None = None,
        condition_dict: Dict[str, Dict] | None = None,
        conditions: Dict[Any, Any] | None = None,
        conditions_combined: List[Any] | None = None,
        unique_gene_list: Dict[Any, Any] | None = None,
        shared_gene_list: Dict[Any, Any] | None = None,
        context_tps: List[int] | None = None,
        mapping_dict_path: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        # only set precision for GPU

        set_matmul_precision_for_device(precision)

        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                gene_to_rowid = pickle.load(f)
                # swap key and value
                gene_to_rowid = {v: k for k, v in gene_to_rowid.items()}
        else:
            raise ValueError('mapping_dict_path is None')
        # change to token_id to gene name
        self.pretrained_model = CytoMeister(
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
            encoder_path=encoder_path,
            pos_encoding_mode=pos_encoding_mode,
            condition_dict=condition_dict,
            gene_to_rowid=gene_to_rowid,
            context_mode=context_mode,
            seed=seed,
        )
        self.pos_encoding_mode = pos_encoding_mode
        # load PETRA checkpoint
        if ckpt_masking_path is not None:
            checkpoint = torch.load(ckpt_masking_path, map_location='cpu')
            state_dict_ = modify_ckpt_state_dict(checkpoint, 'transformer.')
            missing, unexpected = self.pretrained_model.load_state_dict(
                state_dict_, strict=False
            )
            # set parameters to not trainable
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            if len(missing) > 1:
                raise Warning(f'Missing keys in state_dict: {missing}')
            if len(unexpected) > 1:
                raise Warning(f'Unexpected keys in state_dict: {unexpected}')
        self.return_rouge_score = return_rouge_score
        self.decoder = CountDecoder(
            pretrained_model=self.pretrained_model,
            loss_mode=loss_mode,
            d_condc=d_condc,
            d_condt=d_condt,
            layer_norm=layer_norm,
            max_seq_length=max_seq_length,
            use_positional_encoding=use_positional_encoding,
            encoder=encoder,
            pos_encoding_mode=pos_encoding_mode,
            add_cell_time=add_cell_time,
            d_model=d_model,
            dropout=dropout,
            pred_tps=pred_tps,
            context_tps=context_tps,
            n_total_tps=n_total_tps,
            n_genes=n_genes,
            seed=seed,
        )
        if ckpt_count_path is not None:
            checkpoint = torch.load(ckpt_count_path, map_location='cpu')

            state_dict_ = modify_ckpt_state_dict(checkpoint, 'decoder.')
            self.decoder.load_state_dict(state_dict_, strict=False)

        self.weight_decay = weight_decay
        self.lr = lr
        self.loss_mode = loss_mode
        self.d_condc = d_condc
        self.d_condt = d_condt
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
        # total_vocab_size = tgt_vocab_size
        self.pred_tps = pred_tps
        if context_tps is None:
            self.total_tps = pred_tps
        else:
            self.total_tps = context_tps + pred_tps
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
            'pred_counts': [],
            'cls_embeddings': [],
            'cell_idx': [],
        }
        if self.return_rouge_score:
            # load rouge
            self.rouge = evaluate.load('rouge')
            # initiate rouge dict
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

        self.condition_dict = condition_dict

    def forward(self, batch, generate: bool = False):
        tgt_input_id_dict = {}
        for i in self.total_tps:
            tgt_input_id_ = batch[f'tgt_input_ids_t{i}'].clone()
            if self.condition_dict is not None:
                cond_ids = concat_cond_tokens(
                    batch=batch,
                    time_step=i,
                    condition_dict=self.condition_dict,
                )
                tgt_input_id_ = torch.cat((cond_ids, tgt_input_id_), dim=1)
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if generate:
            outputs = None
        else:
            outputs = self.decoder(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
            )

        return outputs, tgt_input_id_dict

    def one_hot_encoder(self, idx, n_cls):
        assert torch.max(idx).item() < n_cls
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        onehot = torch.zeros(idx.size(0), n_cls)
        onehot = onehot.to(idx.device)
        onehot.scatter_(1, idx.long(), 1)
        return onehot

    def compute_dispersion(self, batch):
        dispersions = F.linear(
            self.one_hot_encoder(
                batch['combined_batch'],
                self.n_conditions_combined,
                # self.theta.dtype,
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
                        torch.manual_seed(42)
                        x_pred = zinb_distribution.sample((n_samples,))
                        count_dict[time_step] = x_pred.mean(dim=0)

                elif self.loss_mode == 'nb':
                    nb_distribution = NegativeBinomial(mu=dec_mean, theta=dispersion)
                    loss = -nb_distribution.log_prob(true_values).sum(dim=-1).mean()
                    if n_samples == 1:
                        count_dict[time_step] = dec_mean
                    else:
                        torch.manual_seed(42)
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

    def training_step(self, batch, *args, **kwargs):
        outputs, _ = self.forward(batch)
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
            # mean_pearson = pearson(pred_counts=pred_counts, true_counts=true_counts)
            # random sample 10000 or max number of samples
            if len(pred_counts) > 10000:
                random_ids = torch.randint(
                    low=0,
                    high=len(pred_counts),
                    size=(10000,),
                    generator=torch.Generator().manual_seed(42),
                ).tolist()
            else:
                random_ids = torch.arange(len(pred_counts)).tolist()
            pred_counts = pred_counts[random_ids]
            true_counts = true_counts[random_ids]
            emd = compute_emd(pred_counts, true_counts)
            self.log(
                'train/emd',
                emd,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            # set to status quo
            self.train_dict['true_counts'] = []
            self.train_dict['pred_counts'] = []

    def validation_step(self, batch, *args, **kwargs):
        outputs, _ = self.forward(batch)
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
        with torch.no_grad():
            # return Pearson correlation coefficient
            true_counts = torch.cat(self.val_dict['true_counts'])
            pred_counts = torch.cat(self.val_dict['pred_counts'])
            # mean_pearson = pearson(pred_counts=pred_counts, true_counts=true_counts)
            # random sample 10000 or max number of samples
            if len(pred_counts) > 10000:
                random_ids = torch.randint(
                    low=0,
                    high=len(pred_counts),
                    size=(10000,),
                    generator=torch.Generator().manual_seed(42),
                ).tolist()
            else:
                random_ids = torch.arange(len(pred_counts)).tolist()
            pred_counts = pred_counts[random_ids]
            true_counts = true_counts[random_ids]
            emd = compute_emd(pred_counts, true_counts)
            self.log(
                'val/emd',
                emd,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        self.val_dict['true_counts'] = []
        self.val_dict['pred_counts'] = []

    def test_step(self, batch, *args, **kwargs):
        outputs, tgt_input_id_dict = self.forward(
            batch,
            generate=self.generate,
        )
        if self.condition_dict is not None:
            cond_length = len(self.condition_dict)

        else:
            cond_length = 0
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

            outputs, pred_ids_dict = self.decoder.generate_counts(
                cond_length=cond_length,
                **decoder_kwargs,
            )

            for t in self.pred_tps:
                if self.return_rouge_score:
                    pred_ids = (
                        pred_ids_dict[f'tgt_input_ids_t{t}'].detach().cpu().numpy()
                    )
                    tgt_ids = batch[f'tgt_input_ids_t{t}'].detach().cpu().numpy()
                    test_dict = compute_rouge_score(
                        rouge=self.rouge,
                        pred_ids=pred_ids,
                        tgt_ids=tgt_ids,
                        rouge_len_list=self.rouge_seq_len_list,
                        max_seq_length=self.max_seq_length,
                        test_dict=self.test_dict,
                    )
                    self.test_dict = test_dict
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
            pred_adata = return_generation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_generate_adata_'
                    f't{self.pred_tps}_{self.encoder}_s{self.seed}_'
                    f'l{self.loss_mode}_n{self.n_samples}'
                    f'_p{self.pos_encoding_mode}_'
                    f'm{self.mask_scheduler}_s{self.sequence_length}.h5ad'
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
                        rouge_score = np.concatenate(
                            self.test_dict[f'rouge1_{seq_len}']
                        )
                        metric_mean[f'rouge1_{seq_len}'] = np.mean(rouge_score, axis=0)

            metrics = pd.DataFrame(metric_mean, index=[0])
            # add metrics on the gene space
            lin_reg_df = lin_reg_summary(true_adata, pred_adata)
            mmd_df = evaluate_mmd(true_adata, pred_adata, n_cells=10000)
            metrics = pd.concat([metrics, emd_df, lin_reg_df, mmd_df], axis=1)
            metrics.to_csv(
                f'{self.output_dir}/{self.date}_p{self.pos_encoding_mode}_'
                f'm{self.mask_scheduler}_t{self.temperature}_i{self.iterations}'
                f'_s{self.seed}_s{self.sequence_length}'
                f'n{self.n_samples}_metrics.csv'
            )

        else:
            var_dict = {}
            for var in self.var_list:
                var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)

            pred_adata = ad.AnnData(
                X=torch.cat(self.test_dict['pred_counts']).cpu().numpy(), obs=test_obs
            )
            pred_adata.layers['counts'] = (
                torch.cat(self.test_dict['true_counts']).cpu().numpy()
            )
            pred_adata.write_h5ad(f'{self.output_dir}/pred_adata.h5ad')
            # true counts are stored in the 'counts' layer
            true_adata = pred_adata.copy()
            true_adata.X = true_adata.layers['counts']
            # ----------------- calculate metrics -----------------
            # MSE
            lin_reg_df = lin_reg_summary(true_adata, pred_adata)
            # mmd_df = evaluate_mmd(true_adata, pred_adata, n_cells=10000)
            # emd = evaluate_emd(true_adata, pred_adata)
            metric_df = pd.concat([lin_reg_df], axis=1)
            metric_df.to_csv(f'{self.output_dir}/test_metrics.csv')
            # emd['metric'] = 'emd'
            # emd = emd.rename(columns={'emd': 'value'})
            # self.log(
            #     'test/emd',
            #     emd['value'].mean(),
            #     on_epoch=True,
            #     prog_bar=True,
            #     logger=True,
            # )

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
