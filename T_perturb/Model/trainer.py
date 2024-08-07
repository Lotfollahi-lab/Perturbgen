import os
import pickle
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torchmetrics import MeanSquaredError
from torchmetrics.text import Perplexity

from T_perturb.Model.metric import evaluate_emd
from T_perturb.Modules.T_model import CellGen, CountDecoder
from T_perturb.src.losses import mse_loss
from T_perturb.src.utils import (
    compute_cos_similarity,
    modify_ckpt_state_dict,
    pearson,
    return_cos_similarity,
    return_gene_embeddings,
)

# from deepspeed.ops.adam import FusedAdam


if torch.cuda.is_available():
    cuda_device_name = torch.cuda.get_device_name()
    # If the device is an A100, set the precision for matrix multiplication
    if 'A100' in cuda_device_name:
        torch.set_float32_matmul_precision('medium')


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
        lr: float = 1e-3,
        # lr_scheduler_patience: float = 5.0,
        return_embeddings: bool = False,
        generate: bool = False,
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        mode: str = 'GF_fine_tuned',
        var_list: Optional[List[str]] = None,
        gene_names: Optional[List[str]] = None,
        mapping_dict_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.transformer = CellGen(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            mlm_probability=mlm_probability,
            time_steps=time_steps,
            total_time_steps=total_time_steps,
            mode=mode,
        )

        self.masking_loss = nn.CrossEntropyLoss()
        self.timepoint_loss = nn.CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.lr = lr
        # self.lr_scheduler_patience = lr_scheduler_patience
        # self.lr_scheduler_factor = lr_scheduler_factor
        self.perplexity = Perplexity(ignore_index=-100)
        self.mse = MeanSquaredError()
        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                self.subset_tokenid_to_genename = pickle.load(f)
        self.return_embeddings = return_embeddings
        self.generate = generate
        self.tgt_vocab_size = tgt_vocab_size
        self.time_steps = time_steps

        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'cls_embeddings': [],
            'cosine_similarities': [],
            'batch': [],
            'cell_idx': [],
            'gene_embeddings': [],
        }
        if var_list is not None:
            self.var_list = var_list
            for var in self.var_list:
                self.test_dict[var] = []
        else:
            self.var_list = []

        self.marker_genes = None
        self.gene_names = gene_names
        # register buffer for CLS
        total_vocab_size = tgt_vocab_size
        # initialize cls token for all time steps
        for i in range(1, total_time_steps + 1):
            # i-1, as first token is tgt_vocab_size
            self.register_buffer(
                f'cls_token_{str(i)}',
                torch.tensor(
                    [total_vocab_size + (i - 1)],
                    dtype=torch.long,
                ),
            )
            print(f'cls_token_{str(i)}', getattr(self, f'cls_token_{str(i)}'))
        self.output_dir = output_dir
        # create directory if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.date = datetime.now().strftime('%Y%m%d')

    def forward(self, batch):
        tgt_input_id_dict = {}
        for i in self.time_steps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_id_t{i}'] = tgt_input_id_
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            not_masked=self.return_embeddings,
        )
        return outputs

    def configure_optimizers(self):
        parameters = [{'params': self.transformer.parameters(), 'lr': self.lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        # optimizer = FusedAdam(
        #     self.transformer.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        # lr_scheduler = WarmupCosineLR(
        #     optimizer,
        #     total_num_steps=2000,
        #     # mode='min',
        #     warmup_type = 'linear',
        #     # patience=self.lr_scheduler_patience,
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler,
            # 'scheduler_type': 'WarmupCosineLR',
            'monitor': 'train/masking_loss',
        }

    def training_step(self, batch, *args, **kwargs):
        # logits, labels, count_output, count_dropout = self.forward(batch)
        outputs = self.forward(batch)
        dec_logits = outputs['dec_logits']
        # moe_logits = outputs['moe_logits']
        # time_step = outputs['selected_time_step']
        labels = outputs['labels']
        perp = self.perplexity(dec_logits, labels)
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

        return masking_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        dec_logits = outputs['dec_logits']
        # moe_logits = outputs['moe_logits']
        # time_step = outputs['selected_time_step']
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
            for time_step in self.time_steps:
                token_ids = batch[f'tgt_input_ids_t{time_step}']
                cell_ids = batch[f'tgt_cell_idx_t{time_step}']
                (
                    cos_similarity,
                    cls_embeddings,
                    gene_embeddings,
                ) = compute_cos_similarity(
                    outputs=outputs, time_step=time_step, all_time_steps=self.time_steps
                )

                # define marker gene list to extract gene embeddings
                marker_genes = [
                    'IL7R',
                    'CD52',
                    'GIMAP7',
                    'SARAF',
                    'BTG1',
                    'LTB',
                    'CXCR4',
                    'STAT1',
                    'IRF1',
                    'IFIT3',
                    'GBP1',
                    'SYNE2',
                    'SOCS3',
                    'IL4R',
                    'CD69',
                    'MIR155HG',
                    'DDX21',
                    'TNFRSF4',
                    'HSP90AA1',
                    'HSP90AB1',
                    'HSPA8',
                    'TXN',
                    'FABP5',
                    'TUBA1B',
                    'HMGA1',
                    'PCNA',
                    'IL2RA',
                    'BATF',
                    'CD63',
                    'IFITM2',
                    'CORO1B',
                    'ISG15',
                    'ALDOC',
                    'DDIT4',
                    'LGALS1',
                    'S100A4',
                    'S100A6',
                    'VIM',
                    'CD74',
                    'HLA-DRA',
                    'HLA-DRB1',
                ]

                marker_cos_similarity, marker_genes_dict = return_cos_similarity(
                    marker_genes=marker_genes,
                    cos_similarity=cos_similarity,
                    gene_embeddings=gene_embeddings,
                    mapping_dict=self.subset_tokenid_to_genename,
                    token_ids=token_ids,
                )
                marker_gene_embeddings = return_gene_embeddings(
                    marker_genes=marker_genes,
                    gene_embeddings=gene_embeddings,
                    mapping_dict=self.subset_tokenid_to_genename,
                    token_ids=token_ids,
                )

                self.marker_genes = marker_genes_dict
                self.test_dict['true_counts'].append(
                    batch[f'tgt_counts_t{time_step}'].detach().cpu()
                )
                self.test_dict['cls_embeddings'].append(cls_embeddings.detach().cpu())
                self.test_dict['cosine_similarities'].append(
                    marker_cos_similarity.detach().cpu()
                )
                self.test_dict['batch'].append(batch['combined_batch'].detach().cpu())
                self.test_dict['cell_idx'].append(cell_ids)
                self.test_dict['gene_embeddings'].append(
                    marker_gene_embeddings.detach().cpu()
                )
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])

    def on_test_epoch_end(self):
        if self.return_embeddings:
            print('Start saving embeddings -------------------')
            cls_embeddings = torch.cat(self.test_dict['cls_embeddings'])
            true_counts = torch.cat(self.test_dict['true_counts'])
            cosine_similarities = torch.cat(self.test_dict['cosine_similarities'])
            batch = torch.cat(self.test_dict['batch'])
            cell_ids = np.concatenate(self.test_dict['cell_idx'])
            gene_embeddings = torch.cat(self.test_dict['gene_embeddings'])
            if len(self.var_list) > 0:
                var_dict = {}
                for var in self.var_list:
                    var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)
            test_obs['batch'] = np.array(batch)
            test_obs['cell_idx'] = cell_ids
            adata = ad.AnnData(
                X=true_counts.numpy(),
                obs=test_obs,
                obsm={
                    'cls_embeddings': cls_embeddings.numpy(),
                    'gene_embeddings': gene_embeddings.numpy(),
                },
                uns={
                    'marker_genes': self.marker_genes,
                },
            )
            adata.var_names = self.gene_names
            df = pd.DataFrame(
                cosine_similarities.numpy(), columns=self.marker_genes.keys()
            )
            df.index = adata.obs_names
            adata.obsm['cosine_similarity'] = df
            # save anndata
            adata.write_h5ad(
                f'{self.output_dir}/{self.date}'
                f'cls_embeddings_cosine_similarity.h5ad'
            )
            print('End saving embeddings -------------------')


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
        lr_scheduler_patience: float = 1.0,
        # lr_scheduler_factor: float = 0.8,
        ckpt_masking_path: Optional[str] = None,
        ckpt_count_path: Optional[str] = None,
        conditions: Optional[Dict[Any, Any]] = None,
        conditions_combined: Optional[List[Any]] = None,
        dropout: float = 0.0,
        generate: bool = False,
        var_list: List[str] = ['Time_point'],
        tgt_adata: Optional[ad.AnnData] = None,
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
        temperature: float = 2.0,
        iterations: int = 18,
        n_samples: int = 1,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        mask_scheduler: Optional[str] = 'cosine',
        mode: str = 'GF_fine_tuned',
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        pretrained_model = CellGen(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            time_steps=time_steps,
            total_time_steps=total_time_steps,
            mode=mode,
        )
        # load PETRA checkpoint
        if ckpt_masking_path is not None:
            checkpoint = torch.load(ckpt_masking_path, map_location='cpu')
            state_dict_ = modify_ckpt_state_dict(checkpoint, 'transformer.')
            pretrained_model.load_state_dict(state_dict_, strict=False)
            # set parameters to not trainable
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.decoder = CountDecoder(
            pretrained_model=pretrained_model,
            loss_mode=loss_mode,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            dropout=dropout,
            time_steps=time_steps,
            total_time_steps=total_time_steps,
        )

        if ckpt_count_path is not None:
            checkpoint = torch.load(ckpt_count_path, map_location='cpu')

            state_dict_ = modify_ckpt_state_dict(checkpoint, 'decoder.')
            self.decoder.load_state_dict(state_dict_, strict=False)

        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        # self.lr_scheduler_factor = lr_scheduler_factor
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
                torch.randn(tgt_vocab_size - 1, self.n_conditions_combined)
            )
        else:
            self.theta = None

        self.mse = MeanSquaredError()
        total_vocab_size = tgt_vocab_size
        self.time_steps = time_steps
        self.total_time_steps = total_time_steps  # for generation
        for i in range(1, total_time_steps + 1):
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

        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'ctrl_counts': [],
            'pred_counts': [],
            'cls_embeddings': [],
        }
        self.n_samples = n_samples
        if var_list is not None:
            self.var_list = var_list
            for var in self.var_list:
                self.test_dict[var] = []
        else:
            self.var_list = []
        self.output_dir = output_dir
        # create directory if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # create variables based
        # initiate lists to store true, ctrl and pred counts
        self.train_true_counts_list: List[int] = []
        self.train_pred_counts_list: List[int] = []
        self.val_true_counts_list: List[int] = []
        self.val_pred_delta_counts_list: List[int] = []
        self.val_true_delta_counts_list: List[int] = []
        self.val_pred_counts_list: List[int] = []
        self.val_tgt_cell_type_list: List[str] = []
        self.val_tgt_cell_population_list: List[str] = []
        self.val_tgt_donor_list: List[str] = []
        self.mode = mode
        self.seed = seed
        self.date = datetime.now().strftime('%Y%m%d')

    def forward(self, batch):
        tgt_input_id_dict = {}

        for i in self.time_steps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_id_t{i}'] = tgt_input_id_
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
        loss_list = []
        count_dict = {}
        dispersion = (
            self.compute_dispersion(batch) if self.loss_mode in ['zinb', 'nb'] else None
        )
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
            number of samples to draw from distribution for zinb and nb

        Returns:
        --------
        loss: `torch.Tensor`
            loss value
        count_dict: `Dict[str, torch.Tensor]`
            dictionary containing predicted counts
        """
        for time_step in self.time_steps:
            count_ouput = outputs[f'count_output_t{time_step}']
            true_counts = batch[f'tgt_counts_t{time_step}']
            batch_size_factor = batch[f'tgt_size_factor_t{time_step}']

            if self.loss_mode == 'mse':
                # change true counts dtype to count output dtype
                true_counts = true_counts.type(count_ouput['count_lognorm'].dtype)
                loss = (
                    mse_loss(count_ouput['count_lognorm'], true_counts)
                    .sum(dim=-1)
                    .mean()
                    .float()
                )
                count_dict[time_step] = count_ouput['count_lognorm']
            elif self.loss_mode in ['zinb', 'nb']:
                dec_mean_gamma = count_ouput['count_mean']
                dec_mean = dec_mean_gamma * batch_size_factor.unsqueeze(1).expand(
                    dec_mean_gamma.size(0), dec_mean_gamma.size(1)
                )

                if self.loss_mode == 'zinb':
                    dec_dropout = count_ouput['count_dropout']
                    zinb_distribution = ZeroInflatedNegativeBinomial(
                        mu=dec_mean,
                        theta=dispersion,
                        zi_logits=dec_dropout,
                    )
                    loss = -zinb_distribution.log_prob(true_counts).sum(dim=-1).mean()
                    if n_samples == 1:
                        count_dict[time_step] = dec_mean
                    else:
                        # sample from distribution
                        x_pred = zinb_distribution.sample((n_samples,))
                        count_dict[time_step] = x_pred.mean(dim=0)

                elif self.loss_mode == 'nb':
                    nb_distribution = NegativeBinomial(mu=dec_mean, theta=dispersion)
                    loss = -nb_distribution.log_prob(true_counts).sum(dim=-1).mean()
                    if n_samples == 1:
                        count_dict[time_step] = dec_mean
                    else:
                        x_pred = nb_distribution.sample((n_samples,))
                        count_dict[time_step] = x_pred.mean(dim=0)
            loss_list.append(loss)
        loss = torch.sum(torch.stack(loss_list))
        return loss, count_dict

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

        mse_all = []
        for time_step in self.time_steps:
            pred_count = pred_counts_dict[time_step]
            true_count = batch[f'tgt_counts_t{time_step}']
            # MSE
            mse = self.mse(pred_count, true_count)
            mse_all.append(mse)
            # gather for validation step
            self.train_true_counts_list.append(batch[f'tgt_counts_t{time_step}'])
            self.train_pred_counts_list.append(pred_count)

        mean_mse = torch.mean(torch.stack(mse_all))

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
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.train_true_counts_list)
        pred_counts = torch.cat(self.train_pred_counts_list)
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
        # MSE
        mse_all = []
        for time_step in self.time_steps:
            pred_count = pred_counts_dict[time_step]
            true_count = batch[f'tgt_counts_t{time_step}']
            # MSE
            mse = self.mse(pred_count, true_count)
            mse_all.append(mse)
            # gather for validation step
            self.val_true_counts_list.append(batch[f'tgt_counts_t{time_step}'])
            self.val_pred_counts_list.append(pred_count)

        mean_mse = torch.mean(torch.stack(mse_all))
        self.log(
            'val/mse',
            mean_mse,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return count_loss

    def on_validation_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.val_true_counts_list)
        pred_counts = torch.cat(self.val_pred_counts_list)
        mean_pearson = self.pearson(pred_counts=pred_counts, true_counts=true_counts)
        self.log(
            'val/pearson',
            mean_pearson,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.val_true_counts_list = []
        self.val_ctrl_counts_list = []

    def test_step(self, batch, *args, **kwargs):
        tgt_input_id_dict = {}
        for i in range(1, self.total_time_steps + 1):
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_id_t{i}'] = tgt_input_id_
        interval = batch[f'tgt_input_ids_t{i}'].shape[1] + 1  # as 0 is cls token
        num_steps = len(self.time_steps)
        cls_positions = np.arange(0, num_steps * interval, interval)
        if self.generate:
            outputs = self.decoder.generate(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                max_len=self.max_seq_length,
                mask_scheduler=self.mask_scheduler,
                can_remask_prev_masked=False,
                topk_filter_thres=0.9,
                # time_steps=self.time_steps,
                temperature=self.temperature,
                iterations=self.iterations,
                cls_positions=cls_positions,
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
                batch_size=batch[f'tgt_input_ids_t{self.time_steps[0]}'].shape[0],
            )
            mse_all = []
            for time_step in self.time_steps:
                pred_count = pred_counts_dict[time_step]
                true_count = batch[f'tgt_counts_t{time_step}']
                # MSE
                mse = self.mse(pred_count, true_count)
                mse_all.append(mse)
                # gather for validation step
                self.test_dict['pred_counts'].append(pred_count)
                self.test_dict['true_counts'].append(true_count)
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])
                cls_embeddings = outputs[f'cls_embedding_t{time_step}']
                self.test_dict['cls_embeddings'].append(cls_embeddings)

            mean_mse = torch.mean(torch.stack(mse_all))
            self.log(
                'test/mse',
                mean_mse,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        else:
            outputs = self.forward(batch)
            count_loss, pred_count = self.compute_count_loss(outputs, batch)
            self.log(
                'test/loss',
                count_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['tgt_input_ids'].shape[0],
            )

            self.test_dict['pred_counts'].append(pred_count)
            self.test_dict['true_counts'].append(batch['tgt_counts'])
            self.test_dict['ctrl_counts'].append(batch['src_counts'])
            if len(self.var_list) > 0:
                for var in self.var_list:
                    self.test_dict[var].append(batch[var])

    def on_test_epoch_end(self):
        if self.generate:
            print('---Generating anndata')
            true_counts = torch.cat(self.test_dict['true_counts']).detach().cpu()
            pred_counts = torch.cat(self.test_dict['pred_counts']).detach().cpu()
            # create dict to var_list values
            if len(self.var_list) > 0:
                var_dict = {}
                for var in self.var_list:
                    var_dict[var] = np.concatenate(self.test_dict[var])
                test_obs = pd.DataFrame(var_dict)
            else:
                test_obs = None
            cls_embeddings = torch.cat(self.test_dict['cls_embeddings']).detach().cpu()
            pred_adata = ad.AnnData(X=pred_counts.numpy(), obs=test_obs)
            pred_adata.layers['counts'] = true_counts.numpy()
            pred_adata.obsm['cls_embeddings'] = cls_embeddings.numpy()
            true_adata = pred_adata.copy()
            true_adata.X = true_counts.numpy()
            # create output directory
            # save adata
            pred_adata.write_h5ad(
                f'{self.output_dir}/{self.date}_'
                f'generate_adata_extrapolate_'
                f'{self.time_steps}__{self.mode}_{self.seed}_'
                f'{self.loss_mode}_{self.n_samples}.h5ad'
            )
            print('---anndata generation completed')
            emd = evaluate_emd(true_adata, pred_adata)
            self.log(
                'test/emd',
                emd['emd'].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            # return Pearson correlation coefficient
            true_counts = torch.cat(self.test_true_counts_list)
            pred_counts = torch.cat(self.test_pred_counts_list)
            ctrl_counts = torch.cat(self.test_ctrl_counts_list)
            var_dict = {}
            for var in self.var_list:
                var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)
            pred_adata = ad.AnnData(
                X=pred_counts.numpy(), obs=test_obs, var=self.adata.var
            )
            pred_adata.layers['counts'] = true_counts.numpy()
            pred_adata.write_h5ad(f'{self.output_dir}/pred_adata.h5ad')
            # ----------------- calculate metrics -----------------
            mean_pearson = self.pearson(pred_counts, true_counts)
            # Pearson correlation coefficient
            self.log(
                'test/pearson',
                mean_pearson,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # Pearson delta
            mean_pearson_delta = self.pearson(pred_counts, true_counts, ctrl_counts)
            self.log(
                'test/pearson_delta',
                mean_pearson_delta,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # MSE
            mse = self.metric['mse'](pred_counts, true_counts)

            metrics = pd.DataFrame(
                {
                    'pearson': [mean_pearson.cpu().detach().numpy()],
                    'pearson_delta': [mean_pearson_delta.cpu().detach().numpy()],
                    'mse': [mse.cpu().detach().numpy()],
                }
            )
            metrics.to_csv(f'{self.output_dir}/test_metrics.csv')
            # # calculate MMD and EMD
            # mmd = evaluate_mmd(self.adata, pred_adata)
            # mmd['metric'] = 'mmd'
            # # rename column called mmd
            # mmd = mmd.rename(columns={'mmd': 'value'})
            emd = evaluate_emd(self.adata, pred_adata)
            emd['metric'] = 'emd'
            emd = emd.rename(columns={'emd': 'value'})
            self.log(
                'test/emd',
                emd['value'].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # # concatenate
            # metrics = pd.concat([mmd, emd])
            # # save metrics
            # metrics.to_csv(
            #     f'{self.output_dir}/test_mmd_emd_{condition_key}_metrics.csv'
            # )
            # set to status quo
            self.test_true_counts_list = []
            self.test_ctrl_counts_list = []
            self.test_pred_counts_list = []

    def configure_optimizers(self):
        # optimizer = FusedAdam(
        #     self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        parameters = [{'params': self.decoder.parameters(), 'lr': self.lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        # lr_scheduler = WarmupCosineLR(
        #     optimizer,
        #     total_num_steps=2000,
        #     # mode='min',
        #     warmup_type = 'linear',
        #     # patience=self.lr_scheduler_patience,
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler,
            # 'scheduler_type': 'WarmupCosineLR',
            'monitor': 'train/loss',
        }
