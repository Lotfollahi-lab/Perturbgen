import os
import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import anndata as ad
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
from torchmetrics import (
    CosineSimilarity,
    MeanSquaredError,
    PearsonCorrCoef,
)
from torchmetrics.text import Perplexity

from T_perturb.Model.metric import evaluate_emd, evaluate_mmd  # pearson,
from T_perturb.Modules.T_model import CountDecoder, Petra
from T_perturb.src.losses import mse_loss

# from deepspeed.ops.adam import FusedAdam


if torch.cuda.is_available():
    cuda_device_name = torch.cuda.get_device_name()
    # If the device is an A100, set the precision for matrix multiplication
    if 'A100' in cuda_device_name:
        torch.set_float32_matmul_precision('medium')

sc.settings.set_figure_params(dpi=500)


def batch_token_ranking(tokenised_cells: torch.tensor, vocab_size: int):
    batch_size, seq_length = tokenised_cells.shape

    # tensor of size (batch_size, seq_length) filled with 0 to seq_length
    indices = torch.arange(seq_length, device=tokenised_cells.device).expand(
        batch_size, -1
    )

    # tensor of size (batch_size, vocab_size) filled with seq_length
    first_occurrence_indices = torch.full(
        (batch_size, vocab_size),
        seq_length,
        dtype=torch.long,
        device=tokenised_cells.device,
    )

    # For each vocab item, find the first occurrence index
    for i in range(vocab_size):
        mask = tokenised_cells == i
        valid_indices = torch.where(mask, indices, seq_length)
        first_occurrence_indices[:, i] = valid_indices.min(dim=-1).values

    return first_occurrence_indices


class Petratrainer(LightningModule):
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
        mapping_dict_path: str = (
            './T_perturb/T_perturb/pp/res/eb/token_id_to_genename_all.pkl'
        ),
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        var_list: List[str] = ['Time_point'],
        mode: str = 'GF_fine_tuned',
        gene_names: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.transformer = Petra(
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
        self.metric = nn.ModuleDict(
            {
                'cosine_similarity': CosineSimilarity(reduction='mean'),
                'mse': MeanSquaredError(),
                # 'rmse': MeanSquaredError(squared=False),
                # 'spearman': SpearmanCorrCoef(num_outputs=batch_size),
            }
        )

        with open(
            mapping_dict_path,
            'rb',
        ) as f:
            self.subset_tokenid_to_genename = pickle.load(f)
        self.return_embeddings = return_embeddings
        self.generate = generate
        self.tgt_vocab_size = tgt_vocab_size
        self.time_steps = time_steps
        self.var_list = var_list
        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'cls_embeddings': [],
            'cosine_similarities': [],
            'batch': [],
            'cell_idx': [],
            'gene_embeddings': [],
        }
        for var in self.var_list:
            self.test_dict[var] = []

        self.marker_genes = None
        self.gene_names = gene_names
        self.activation_genes = None
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
        interval = batch[f'tgt_input_ids_t{i}'].shape[1] + 1  # as 0 is cls token
        num_steps = len(self.time_steps)
        cls_positions = np.arange(0, num_steps * interval, interval)
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            cls_positions=cls_positions,
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
        # self.log(
        #     'train/moe_loss',
        #     (1 - self.alpha) * moe_loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=batch['tgt_input_ids_t1'].shape[0],
        #     rank_zero_only=True,
        #     sync_dist=True,
        # )
        # self.log(
        #     'train/combined_loss',
        #     combined_loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=batch['tgt_input_ids_t1'].shape[0],
        #     rank_zero_only=True,
        #     sync_dist=True,
        # )

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
        # return F1 score and accuracy
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
            for i, time_step in enumerate(self.time_steps):
                cls_position = outputs['cls_positions'][i]
                cls_embeddings = outputs['dec_embedding'][:, cls_position, :]
                token_ids = batch[f'tgt_input_ids_t{time_step}']
                cell_ids = batch[f'tgt_cell_idx_t{time_step}']
                # exclude cls token from gene embeddings
                if time_step == max(self.time_steps):
                    gene_embeddings = outputs['dec_embedding'][
                        :, (cls_position + 1) :, :
                    ]
                else:
                    gene_embeddings = outputs['dec_embedding'][
                        :, (cls_position + 1) : outputs['cls_positions'][time_step], :
                    ]
                cosine_similarity_list = []
                for i in range(gene_embeddings.shape[0]):
                    # print(gene_embeddings[i, :, :])
                    # print(gene_embeddings[i, :, :].shape)
                    # gene level cosine similarity
                    tmp_consine_similarity = F.cosine_similarity(
                        cls_embeddings[i],
                        gene_embeddings[i, :, :],
                        dim=1,
                    )
                    cosine_similarity_list.append(tmp_consine_similarity)
                cosine_similarity_list = torch.stack(cosine_similarity_list)
                # print('dict_token_mapping',dict_token_mapping)

                # extra
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

                # filter for marker genes and swap key value
                marker_genes_ids = {
                    v: k
                    for k, v in self.subset_tokenid_to_genename.items()
                    if v in marker_genes
                }
                # print('token_dict',self.subset_tokenid_to_genename)
                self.marker_genes = marker_genes_ids

                emb = torch.zeros(
                    cosine_similarity_list.shape[0],
                    len(marker_genes_ids.keys()),
                    device=gene_embeddings.device,
                )
                for i, gene in enumerate(marker_genes_ids.keys()):
                    cond_embs_to_fill = (token_ids == marker_genes_ids[gene]).sum(1) > 0
                    cond_select_markers = torch.where(
                        token_ids == marker_genes_ids[gene]
                    )
                    emb[cond_embs_to_fill, i] = cosine_similarity_list[
                        cond_select_markers[0], cond_select_markers[1]
                    ]

                activation_genes = [
                    'IL7R',
                    'CD69',
                    'IL2RA',
                    'ISG15',
                ]
                # extract gene embedding for activation genes
                activation_genes_ids = {
                    v: k
                    for k, v in self.subset_tokenid_to_genename.items()
                    if v in activation_genes
                }
                activation_gene_embeddings = torch.zeros(
                    gene_embeddings.shape[0],
                    len(activation_genes_ids.keys()),
                    gene_embeddings.shape[2],
                    device=gene_embeddings.device,
                )
                activation_genes_dict = {}
                for i, gene in enumerate(activation_genes_ids.keys()):
                    cond_embs_to_fill = (token_ids == activation_genes_ids[gene]).sum(
                        1
                    ) > 0
                    cond_select_markers = torch.where(
                        token_ids == activation_genes_ids[gene]
                    )
                    activation_gene_embeddings[cond_embs_to_fill, i] = gene_embeddings[
                        cond_select_markers[0], cond_select_markers[1], :
                    ]
                    activation_genes_dict[gene] = i
                self.activation_genes = activation_genes_dict

                self.test_dict['true_counts'].append(
                    batch[f'tgt_counts_t{time_step}'].detach().cpu()
                )
                self.test_dict['cls_embeddings'].append(cls_embeddings.detach().cpu())
                self.test_dict['cosine_similarities'].append(emb.detach().cpu())
                self.test_dict['batch'].append(batch['combined_batch'].detach().cpu())
                self.test_dict['cell_idx'].append(cell_ids)
                self.test_dict['gene_embeddings'].append(
                    activation_gene_embeddings.detach().cpu()
                )
                for var in self.var_list:
                    self.test_dict[var].append(batch[f'{var}_t{time_step}'])

            # self.tgt_output['cell_type'].append(batch['tgt_cell_type'])
            # self.tgt_output['cell_population'].append(batch['tgt_cell_population'])
            # self.time_point = batch['tgt_time_point']

    def on_test_epoch_end(self):
        if self.return_embeddings:
            print('Start saving embeddings -------------------')
            cls_embeddings = torch.cat(self.test_dict['cls_embeddings'])
            true_counts = torch.cat(self.test_dict['true_counts'])
            cosine_similarities = torch.cat(self.test_dict['cosine_similarities'])
            batch = torch.cat(self.test_dict['batch'])
            cell_ids = np.concatenate(self.test_dict['cell_idx'])
            gene_embeddings = torch.cat(self.test_dict['gene_embeddings'])
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
                    'activation_genes': self.activation_genes,
                },
            )
            adata.var_names = self.gene_names
            # self.adata.obsm[marker_genes[i]] = emb.numpy()
            # create a dataframe and annotate columns as marker genes
            df = pd.DataFrame(
                cosine_similarities.numpy(), columns=self.marker_genes.keys()
            )

            df.index = adata.obs_names
            adata.obsm['cosine_similarity'] = df
            # save anndata
            adata.write_h5ad(f'{self.output_dir}/cls_embeddings_cosine_similarity.h5ad')
            print('End saving embeddings -------------------')


class CountDecodertrainer(LightningModule):
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        pretrained_model = Petra(
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
            state_dict_ = self.modify_ckpt_state_dict(checkpoint, 'transformer.')
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

            state_dict_ = self.modify_ckpt_state_dict(checkpoint, 'decoder.')
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

        self.metric = nn.ModuleDict({'mse': MeanSquaredError()})
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
        self.var_list = var_list
        for var in self.var_list:
            self.test_dict[var] = []
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

    def modify_ckpt_state_dict(
        self,
        checkpoint,
        replace_str,
    ):
        if 'module' in checkpoint.keys():
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(replace_str):
                k = k.replace(replace_str, '', 1)
            new_state_dict[k] = v

        return new_state_dict

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
        interval = batch[f'tgt_input_ids_t{i}'].shape[1] + 1  # as 0 is cls token
        num_steps = len(self.time_steps)
        cls_positions = np.arange(0, num_steps * interval, interval)

        outputs = self.decoder(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            cls_positions=cls_positions,
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

    @staticmethod
    def pearson(
        pred_counts: torch.Tensor,
        true_counts: torch.Tensor,
        ctrl_counts: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Pearson correlation coefficient
        """
        if ctrl_counts is not None:
            pred_counts = pred_counts - ctrl_counts
            true_counts = true_counts - ctrl_counts
        num_outputs = true_counts.shape[0]
        pearson = PearsonCorrCoef(num_outputs=num_outputs).to('cuda')
        pred_counts_t = pred_counts.transpose(0, 1)
        true_counts_t = true_counts.transpose(0, 1)
        pearson_output = pearson(pred_counts_t, true_counts_t)
        mean_pearson = torch.mean(pearson_output)
        return mean_pearson

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
            mse = self.metric['mse'](pred_count, true_count)
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

        # RMSE
        # rmse=self.metric['rmse'](outputs['count_mean'], batch['tgt_counts'])
        # mean_rmse = torch.mean(rmse)
        # self.log(
        #     'train/rmse',
        #     mean_rmse,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        return count_loss

    def on_train_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.train_true_counts_list)
        pred_counts = torch.cat(self.train_pred_counts_list)
        # Pearson correlation coefficient
        mean_pearson = self.pearson(pred_counts=pred_counts, true_counts=true_counts)
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
            mse = self.metric['mse'](pred_count, true_count)
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

        # create adata for mmd and emd
        if self.generate:
            cell_type = np.concatenate(self.val_tgt_cell_type_list)
            cell_population = np.concatenate(self.val_tgt_cell_population_list)
            tgt_donor = np.concatenate(self.val_tgt_donor_list)
            true_counts = true_counts.detach().cpu()
            pred_counts = pred_counts.detach().cpu()
            test_obs = pd.DataFrame(
                np.array([cell_type, cell_population, tgt_donor]).T,
                columns=['Cell_type', 'Cell_population', 'Donor'],
            )
            pred_adata = ad.AnnData(X=pred_counts.numpy(), obs=test_obs)
            true_adata = ad.AnnData(X=true_counts.numpy(), obs=test_obs)
            # create mock column with the same value
            pred_adata.obs['emd_tmp'] = 0
            true_adata.obs['emd_tmp'] = 0
            # calculate emd
            emd = evaluate_emd(true_adata, pred_adata)
            # get the value for loggin
            print('EMD:', emd['emd'].item())
            self.log(
                'val/emd',
                emd['emd'].item(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            mmd = evaluate_mmd(true_adata, pred_adata, 'Cell_population')
            # get mean of mmd
            print('MMD:', mmd['mmd'].mean().item())
            self.log(
                'val/mmd',
                mmd['mmd'].mean().item(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        else:
            mean_pearson = self.pearson(
                pred_counts=pred_counts, true_counts=true_counts
            )
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
        self.val_pred_counts_list = []
        self.val_tgt_cell_type_list = []
        self.val_tgt_cell_population_list = []
        self.val_tgt_donor_list = []

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
                mse = self.metric['mse'](pred_count, true_count)
                mse_all.append(mse)
                # gather for validation step
                self.test_dict['pred_counts'].append(pred_count)
                self.test_dict['true_counts'].append(true_count)
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
            for var in self.var_list:
                self.test_dict[var].append(batch[var])

    def on_test_epoch_end(self):
        if self.generate:
            print('---Generating anndata')
            true_counts = torch.cat(self.test_dict['true_counts']).detach().cpu()
            pred_counts = torch.cat(self.test_dict['pred_counts']).detach().cpu()
            # create dict to var_list values
            var_dict = {}
            for var in self.var_list:
                var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)
            cls_embeddings = torch.cat(self.test_dict['cls_embeddings']).detach().cpu()
            # test_obs = pd.DataFrame(
            #     np.array(
            #         [tgt_cell_type, tgt_cell_population, tgt_time_point, tgt_donor]
            #     ).T,
            #     columns=['Cell_type', 'Cell_population', 'Time_point', 'Donor'],
            # )
            pred_adata = ad.AnnData(X=pred_counts.numpy(), obs=test_obs)
            pred_adata.layers['counts'] = true_counts.numpy()
            pred_adata.obsm['cls_embeddings'] = cls_embeddings.numpy()
            true_adata = pred_adata.copy()
            true_adata.X = true_counts.numpy()
            # create output directory
            # save adata
            pred_adata.write_h5ad(
                f'{self.output_dir}/generate_adata_extrapolate_t4_{self.mode}'
                f'_{self.loss_mode}_{self.n_samples}.h5ad'
            )
            emd = evaluate_emd(true_adata, pred_adata)
            self.log(
                'test/emd',
                emd['emd'].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # mmd = evaluate_mmd(
            #     adata=true_adata,
            #     pred_adata=pred_adata,
            #     n_cells=5000,
            # )
            # self.log(
            #     'test/mmd',
            #     mmd['mmd'].mean(),
            #     on_epoch=True,
            #     prog_bar=True,
            #     logger=True,
            # )
            print('---anndata generation completed')
        else:
            # return Pearson correlation coefficient
            true_counts = torch.cat(self.test_true_counts_list)
            pred_counts = torch.cat(self.test_pred_counts_list)
            ctrl_counts = torch.cat(self.test_ctrl_counts_list)
            tgt_cell_type = np.concatenate(self.test_tgt_cell_type_list)
            tgt_cell_population = np.concatenate(self.test_tgt_cell_population_list)
            tgt_donor = np.concatenate(self.test_tgt_donor_list)
            test_obs = pd.DataFrame(
                np.array([tgt_cell_type, tgt_cell_population, tgt_donor]).T,
                columns=['Cell_type', 'Cell_population', 'Donor'],
            )
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
