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
from T_perturb.src.utils import (
    WarmupScheduler,
    compute_cos_similarity,
    modify_ckpt_state_dict,
    return_prediction_adata,
    pearson,
    return_cos_similarity,
    return_gene_embeddings,
    WarmupScheduler
)

# from deepspeed.ops.adam import FusedAdam


def set_matmul_precision_for_device():
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name()
    # If the device is an A100, set the precision for matrix multiplication
    if ('A100' in cuda_device_name) or ('NVIDIA H100 80GB HBM' in cuda_device_name):
        print(f'Using {cuda_device_name} for training')
        print('Set float32_matmul_precision to medium')
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
        initial_lr: float = 1e-4,
        end_lr: float = 1e-3,
        # lr_scheduler_patience: float = 5.0,
        return_embeddings: bool = False,
        generate: bool = False,
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
        num_epochs: int = 5,
        warmup_epochs: int = 1,
        output_dir: str = '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/',
        mode: str = 'GF_fine_tuned',
        mask_scheduler: str = 'cosine',
        context_mode: bool = True,
        positional_encoding: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt'
        ] = 'time_pos_sin',
        var_list: Optional[List[str]] = None,
        gene_names: Optional[List[str]] = None,
        mapping_dict_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        set_matmul_precision_for_device()
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
            #mask_scheduler=mask_scheduler,
            position_embedding=positional_encoding,
        )
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.masking_loss = nn.CrossEntropyLoss()
        self.timepoint_loss = nn.CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.initial_lr = initial_lr 
        self.end_lr = end_lr
        # self.lr_scheduler_patience = lr_scheduler_patience
        # self.lr_scheduler_factor = lr_scheduler_factor
        self.perplexity = Perplexity(ignore_index=-100)
        self.mse = MeanSquaredError()
        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                gene_to_tokenid = pickle.load(f)
        # change to token_id to gene name
        self.gene_to_tokenid = gene_to_tokenid
        self.return_embeddings = return_embeddings
        self.generate = generate
        self.tgt_vocab_size = tgt_vocab_size
        self.time_steps = time_steps
        self.context_mode = context_mode

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
        total_vocab_size = tgt_vocab_size
        # register buffer for CLS
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
        self.date = datetime.now().strftime('%Y%m%d-%H:%M')

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
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            not_masked=self.return_embeddings,
            context_mode=self.context_mode,
        )
        return outputs

    def configure_optimizers(self):
        parameters = [{'params': self.transformer.parameters(), 'lr': self.end_lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        # Calculate total steps and warmup steps based on number_of_batches_per_epoch
        number_of_batches_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.num_epochs * number_of_batches_per_epoch
        warmup_steps = self.warmup_epochs * number_of_batches_per_epoch
        scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, initial_lr=self.initial_lr, end_lr=self.end_lr)
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
            'lr_scheduler': scheduler,
            # 'scheduler_type': 'WarmupCosineLR',
            'monitor': 'train/masking_loss',
            'interval': 'step',
        }

    def training_step(self, batch, *args, **kwargs):
        # logits, labels, count_output, count_dropout = self.forward(batch)
        outputs = self.forward(batch)
        dec_logits = outputs['dec_logits']
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
                #cell_ids = batch[f'tgt_cell_idx_t{time_step}']
                (
                    cos_similarity,
                    cls_embeddings,
                    gene_embeddings,
                ) = compute_cos_similarity(
                    outputs=outputs, time_step=time_step
                )
                input_path = "/lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/top_2000_genes_90m.pkl"
                # Load the dictionary from the file
                with open(input_path, 'rb') as f:
                    marker_genes = pickle.load(f)

                # Use all genes instead of marker genes
                # all_genes = list(self.token_id_to_ensembl.values())
                # time 1 and 3
                #marker_genes = ['X1K', 'FRCL1', 'PCYT1B', 'THCYT2', 'LEFTYB', 'PPP1R146', 
                #                'TMEM17', '12S-LOX', 'M1AP', 'PSS3', 'BA3L8.2', 'SLC59A2', 'URG', 'PTGES3L', 
                #                'RINGO2', 'HSPTB1', 'FAM26E', 'GSHPX-GI', 'CTTN', 'OTR', 'SLFN14', 'IRXB1', 
                #                'CHRNA2', 'CALPAIN11', 'CD42B-ALPHA', 'PF22', 'OX4OL', 'CKLFSF1', 'HG1H', 
                #                'CFAP161', 'NH32', 'JEDI', 'MYEOV', 'FAM81B', 'DYTN', 'LGALSL', 'GFI1B', 
                #                'ENKUR', 'MU1B', 'TMEM158', 'CD240D', 'TAL-1', 'SYTL4', 'AQP10', 'C12ORF39', 
                #                'FAP50', 'GPIIIA', 'NYD-SP9', 'IA-2/PTP', 'SAMD14']
                # 90 min
                #marker_genes = ['CNK3', 'FGF2', 'PLC-L3', 'CYK8', 'DYNC1I1', 'WFDC3', 'SARG', 'PSS3', 'C1QB', 'PLCD4', 'GABRR2', 'SLC24A3', 'GPR91', 'AVPR1', 'IRXB1', 'C20ORF26', 'C1ORF150', 'PLCA2', 'BRGDA9', 'DMD', 'ILDR2', 'C1ORF49', 'HWNT11', 'SEP', 'PCYT1B', 'CHRM3', 'GAREM1', 'C6ORF206', 'TMEM158', 'HYSP1', 'SOX7', 'CKLFSF1', 'CR1L', 'TMPRSS9', 'GPR152', 'SLFN14', 'PF22', 'AGP-A', 'R-PTP-S', 'X1K', 'SCA41', 'GCPS', 'CGSPDE', 'ASAP2', 'LINC00889', 'TRP6', 'MATN2', 'PLTP', 'AMOTL1', 'C10ORF47']
                # 10 hours
                #marker_genes = [['SLM-2', 'EJM7', 'PRELID4B', 'IDAX', 'TAAL6', 'DYTN', 'TMEM158', 'PCYT1B', '447AA', 'MIXL1', 'APCL', 'C1QTNF4', 'GSTM5', 'VEJAM', 'HYSP1', 'ZNF878', 'NKX6-3', 'CD11D', 'HSPTB1', 'BAALC', 'CLEC2L', 'NCX3', 'GLUK4-2', 'SAMD14', 'FRCL1', 'PPP1R146', 'IRXB1', 'CHRNA2', 'SMTCK', 'LEFTYB', 'ABLIM3', 'CRYM', 'CMS19', 'TRP6', 'SLFN14', 'PLA2G2D', 'HJURP', 'SLC59A2', 'TNCY', 'MOX', 'PF22', 'IA-2/PTP', 'GUC2D', 'BCL2L9', 'PFN4', 'DEE101', 'URG', 'DR6', 'THOX1', 'EST155051']]
                #marker_cos_similarity, marker_genes_dict = return_cos_similarity(
                #    cos_similarity=cos_similarity,
                #    gene_embeddings=gene_embeddings,
                #    mapping_dict=self.gene_to_tokenid,
                #    token_ids=token_ids,
                #)
                #print('All genes dict', all_genes_dict)
                #print('cosine similarity', all_gene_cos_similarity)
                #print(marker_cos_similarity.nonzero().size(0))
                #print(marker_cos_similarity.shape)

                marker_gene_embeddings = return_gene_embeddings(
                    marker_genes=marker_genes,  # updated argument name
                    gene_embeddings=gene_embeddings,
                    mapping_dict=self.gene_to_tokenid,
                    token_ids=token_ids,
                )
                ##self.marker_genes = marker_genes_dict
                self.marker_genes = marker_genes
                #print('Marker Genes or marker_genes',self.marker_genes)

                true_counts = batch[f'tgt_counts_t{time_step}'].detach().cpu()
                cls_embeddings = cls_embeddings.detach().cpu()
                #cos_similarity = marker_cos_similarity.detach().cpu()
                gene_embeddings = marker_gene_embeddings.detach().cpu()
                combined_batch = batch['combined_batch'].detach().cpu()
                self.test_dict['true_counts'].append(true_counts)
                self.test_dict['cls_embeddings'].append(cls_embeddings)
                ##self.test_dict['cosine_similarities'].append(cos_similarity)
                self.test_dict['gene_embeddings'].append(gene_embeddings)
                self.test_dict['batch'].append(combined_batch)
                #self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{time_step}'])
                self.test_dict['cell_idx'].append(torch.tensor(batch[f'tgt_cell_idx_t{time_step}']))
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])    
                #print('Test dict', self.test_dict)                

    def on_test_epoch_end(self):
        if self.return_embeddings:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['batch', 'cell_idx'])
            # Load gene names from the specified AnnData file
            adata = sc.read("/lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/T_perturb/pp/res/new_subset_LPS/h5ad_pairing_hvg/LPS_hvg_with_gene_name.h5ad")
            gene_names = adata.var["gene_name"].tolist()  # Convert to list            
            return_prediction_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                marker_genes=self.marker_genes,
                gene_names=gene_names,
                output_dir='/lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/',
                file_name=f'{self.date}_prediction_embeddings_time90m_e19_top2000.h5ad',
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
        lr_scheduler_patience: float = 1.0,
        # lr_scheduler_factor: float = 0.8,
        dropout: float = 0.0,
        generate: bool = False,
        var_list: List[str] = ['Time_point'],
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
        temperature: float = 2.0,
        iterations: int = 18,
        n_samples: int = 1,
        output_dir: str = '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb',
        mode: str = 'GF_fine_tuned',
        mapping_dict_path: str = (
            '/lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/token_dictionary_gc95M.pkl'
        ),
        seed: int = 42,
        context_mode: bool = True,
        n_genes: int = 25426,
        positional_encoding: Literal[
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.save_hyperparameters()
        set_matmul_precision_for_device()
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
            time_steps=time_steps,
            total_time_steps=total_time_steps,
            mode=mode,
            position_embedding=positional_encoding,
        )
        self.positional_encoding = positional_encoding
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
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            dropout=dropout,
            time_steps=time_steps,
            total_time_steps=total_time_steps,
            context_mode=context_mode,
            n_genes=n_genes,
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
                torch.randn(n_genes, self.n_conditions_combined)
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
        self.sequence_length = sequence_length

        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'ctrl_counts': [],
            'pred_counts': [],
            'cls_embeddings': [],
        }
        if self.return_rouge_score:
            self.seq_len_list = [25, 100, max_seq_length]
            for seq_len in self.seq_len_list:
                self.test_dict[f'rouge1_{seq_len}'] = []
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
        self.date = datetime.now().strftime('%Y%m%d-%H:%M')
        # guided generation

        self.unique_gene_list = unique_gene_list
        self.shared_gene_list = shared_gene_list

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
            number of samples to draw from distribution for zinb and nb

        Returns:
        --------
        loss: `torch.Tensor`
            loss value
        count_dict: `Dict[str, torch.Tensor]`
            dictionary containing predicted counts
        """
        loss_list = []
        count_dict = {}
        dispersion = (
            self.compute_dispersion(batch) if self.loss_mode in ['zinb', 'nb'] else None
        )

        for time_step in self.time_steps:
            count_ouput = outputs[f'count_output_t{time_step}']
            true_values = batch[f'tgt_counts_t{time_step}']
            batch_size_factor = batch[f'tgt_size_factor_t{time_step}']

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
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if self.generate:
            decoder_args = {
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
                decoder_args['unique_gene_list'] = self.unique_gene_list
                decoder_args['shared_gene_list'] = self.shared_gene_list
                outputs, pred_ids_dict = self.decoder.guided_generate(
                    **decoder_args,
                )
            else:
                outputs, pred_ids_dict = self.decoder.generate(
                    **decoder_args,
                )
            # print(pred_ids_dict)
            for time_step in pred_ids_dict.keys():
                if self.return_rouge_score:
                    pred_ids = pred_ids_dict[time_step].cpu().numpy()
                    tgt_ids = batch[time_step].cpu().numpy()
                    # exclude task token and padding token
                    # exclude padding token
                    pred_ids = pred_ids.astype(object)
                    pred_ids = pred_ids[:, 1:]  # exclude task token
                    tgt_ids = tgt_ids.astype(object)
                    special_tokens = np.array([0, 1, 2, 3])
                    pred_ids[np.isin(pred_ids, special_tokens)] = ''
                    tgt_ids[np.isin(tgt_ids, special_tokens)] = ''

                    def map_token_to_ensembl(val):
                        return self.token_id_to_ensembl.get(
                            val, val
                        )  # Return mapped value, or original if not in dict

                    # Vectorize the function to apply to the entire matrix
                    vectorized_map = np.vectorize(map_token_to_ensembl)
                    # Apply the mapping
                    pred_ids_ = vectorized_map(pred_ids)
                    tgt_ids_ = vectorized_map(tgt_ids)
                    for seq_len in self.seq_len_list:
                        if self.max_seq_length > seq_len:
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
                        pred_ids_str = np.char.rstrip(pred_ids_str)
                        tgt_ids_str = np.char.rstrip(tgt_ids_str)
                        # create a list of strings
                        pred_ids_str = pred_ids_str.tolist()
                        tgt_ids_str = tgt_ids_str.tolist()

                        # compute rouge score
                        rouge_score = self.rouge.compute(
                            predictions=pred_ids_str,
                            references=tgt_ids_str,
                            rouge_types=['rouge1'],
                        )
                        self.test_dict[f'rouge1_{seq_len}'].append(
                            rouge_score['rouge1']
                        )
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
            for time_step in self.time_steps:
                self.test_dict['pred_counts'].append(pred_count[time_step])
                self.test_dict['true_counts'].append(batch[f'tgt_counts_t{time_step}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])

    def on_test_epoch_end(self):
        # TODO: clean up no if and else needed
        true_counts = torch.cat(self.test_dict['true_counts']).detach().cpu()
        pred_counts = torch.cat(self.test_dict['pred_counts']).detach().cpu()
        if self.generate:
            print('---Generating anndata')

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
            # use scanpy pca to reduce dimensionality

            # create output directory
            # save adata
            pred_adata.write_h5ad(
                f'{self.output_dir}/{self.date}_'
                f'random_pairing_stratified_pairing_generate_adata_'
                f't{self.time_steps}_{self.mode}_s{self.seed}_'
                f'l{self.loss_mode}_n{self.n_samples}'
                f'_p{self.positional_encoding}_'
                f'm{self.mask_scheduler}_s{self.sequence_length}.h5ad'
            )
            print('---anndata generation completed')
            print('---Start saving metrics')
            # save metrics
            metric_mean = {}
            # true counts are stored in the 'counts' layer
            true_adata = pred_adata.copy()
            true_adata.X = true_adata.layers['counts']
            # log norm and compute PCA
            sc.pp.normalize_total(pred_adata, target_sum=1e4)
            sc.pp.log1p(pred_adata)
            sc.tl.pca(pred_adata, svd_solver='arpack', n_comps=5)
            sc.pp.normalize_total(true_adata, target_sum=1e4)
            sc.pp.log1p(true_adata)
            sc.tl.pca(true_adata, svd_solver='arpack', n_comps=5)
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
                if self.test_dict[f'rouge1_{self.seq_len_list[0]}']:
                    for seq_len in self.seq_len_list:
                        metric_mean[f'rouge1_{seq_len}'] = np.mean(
                            self.test_dict[f'rouge1_{seq_len}']
                        )

            metrics = pd.DataFrame(metric_mean, index=[0])
            # add metrics on the gene space
            lin_reg_df = lin_reg_summary(true_adata, pred_adata)
            mmd_df = evaluate_mmd(true_adata, pred_adata, n_cells=10000)
            metrics = pd.concat([metrics, emd_df, lin_reg_df, mmd_df], axis=1)
            metrics.to_csv(
                f'{self.output_dir}/{self.date}_p{self.positional_encoding}_'
                f'm{self.mask_scheduler}_t{self.temperature}_i{self.iterations}'
                '_s{self.sequence_length}_metrics.csv'
            )

        else:
            var_dict = {}
            for var in self.var_list:
                var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)
            pred_adata = ad.AnnData(X=pred_counts.numpy(), obs=test_obs)
            pred_adata.layers['counts'] = true_counts.numpy()
            pred_adata.write_h5ad(f'{self.output_dir}/pred_adata.h5ad')
            # true counts are stored in the 'counts' layer
            true_adata = pred_adata.copy()
            true_adata.X = true_adata.layers['counts']
            # ----------------- calculate metrics -----------------
            # MSE
            lin_reg_df = lin_reg_summary(true_adata, pred_adata)
            mmd_df = evaluate_mmd(true_adata, pred_adata, n_cells=10000)
            emd = evaluate_emd(true_adata, pred_adata)
            metric_df = pd.concat([lin_reg_df, mmd_df, emd], axis=1)
            metric_df.to_csv(f'{self.output_dir}/test_metrics.csv')
            # # calculate MMD and EMD
            # mmd = evaluate_mmd(self.adata, pred_adata)
            # mmd['metric'] = 'mmd'
            # # rename column called mmd
            # mmd = mmd.rename(columns={'mmd': 'value'})

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
