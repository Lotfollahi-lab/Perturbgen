import os
import pickle

# import re
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torchmetrics import MeanSquaredError
from torchmetrics.text import Perplexity, rouge
from tqdm import tqdm

from T_perturb.Model.metric import evaluate_emd  # pearson,
from T_perturb.Modules.T_model import CellGen, CountDecoder
from T_perturb.src.losses import mse_loss
from T_perturb.src.utils import (
    compute_cos_similarity,
    generate_padding,
    gumbel_sample,
    modify_ckpt_state_dict,
    noise_schedule,
    pearson,
    return_cos_similarity,
    return_gene_embeddings,
    top_k,
)

# from deepspeed.ops.adam import FusedAdam


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
        lr: float = 1e-3,
        # lr_scheduler_patience: float = 5.0,
        return_embeddings: bool = False,
        generate: bool = False,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        var_list: List[str] = ['Time_point'],
        encoder_type: Literal[
            'GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'
        ] = 'GF_frozen',
        moe_type: Literal['moe_attention', 'none', 'moe_ffn'] = 'none',
        alpha: float = 0.5,
        n_task_conditions: int = 2,
        gene_names: Optional[List[str]] = None,
        tokenid_to_genename_dict: Optional[str] = None,
        mask_scheduler: Optional[str] = 'cosine',
        temperature: Optional[float] = 2.0,
        iterations: Optional[int] = 18,
        apply_attn_mask: Optional[bool] = False,
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
            encoder_type=encoder_type,
            moe_type=moe_type,
            n_task_conditions=n_task_conditions,
        )
        # if ckpt_masking_path is not None:
        #     print('Start loading checkpoint of masking model')
        #     print(ckpt_masking_path)
        #     checkpoint = torch.load(ckpt_masking_path, map_location='cpu')
        #     self.transformer.load_state_dict(checkpoint, strict=False)
        #     print('Loaded checkpoint of masking model')
        #     pattern = re.compile(r'decoder_block.0.feed_forward.fc2.weight')
        #     state_dict = self.transformer.state_dict()
        #     for name, param in state_dict.items():
        #         if pattern.match(name):
        #             if 'weight' in name:  # This checks if the parameter is a weight
        #                 print(f"{name}: {param.data}")
        #     print('End loading checkpoint of masking model')

        self.masking_loss = nn.CrossEntropyLoss()
        self.timepoint_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha

        self.weight_decay = weight_decay
        self.lr = lr
        # self.lr_scheduler_patience = lr_scheduler_patience
        # self.lr_scheduler_factor = lr_scheduler_factor
        self.perplexity = Perplexity(ignore_index=-100)
        self.rouge = rouge.ROUGEScore(rouge_keys='rouge1')
        self.mse = MeanSquaredError()

        self.return_embeddings = return_embeddings
        self.generate = generate
        self.tgt_vocab_size = tgt_vocab_size
        self.var_list = var_list
        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'cls_embeddings': [],
            'mean_embeddings': [],
            'cosine_similarities': [],
            'batch': [],
            'cell_idx': [],
            'gene_embeddings': [],
            'router_probs': [],
        }
        for var in self.var_list:
            self.test_dict[var] = []
        # total_vocab_size = (
        #     tgt_vocab_size + n_task_conditions
        # )  # add one for each cls token
        self.mask_token = 1  # as defined in Geneformer
        self.marker_genes = None
        self.gene_names = gene_names
        # initialize task token for different conditions (e.g. diseases)
        self.output_dir = output_dir
        # create directory if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.apply_attn_mask = apply_attn_mask
        self.date = datetime.now().strftime('%Y%m%d')
        self.moe_type = moe_type
        if tokenid_to_genename_dict is not None:
            with open(tokenid_to_genename_dict, 'rb') as f:
                self.tokenid_to_genename_dict = pickle.load(f)
        if self.generate:
            self.max_seq_length = max_seq_length
            self.mask_scheduler = mask_scheduler
            self.temperature = temperature
            self.iterations = iterations

    def forward(self, batch):
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id=batch['tgt_input_ids'],
            apply_attn_mask=self.apply_attn_mask,
        )
        # print('Loaded checkpoint of masking model')
        # pattern = re.compile(r'decoder_fc.weight')
        # state_dict = self.transformer.state_dict()
        # for name, param in state_dict.items():
        #     if pattern.match(name):
        #         if 'weight' in name:  # This checks if the parameter is a weight
        #             print(f"{name}: {param.data[:15,:15]}")
        # print('End loading checkpoint of masking model')

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

    @torch.no_grad()
    def iterative_generate(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id: torch.Tensor,
        max_len: int,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        temperature: float = 2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        iterations: int = 18,  # optimal of iterations in MaskGIT
        mask_scheduler: str = 'cosine',
    ):
        '''
        Description:
        ------------
        Generate sequences for the target tokens
        adopted from MaskGIT using the pretrained model.
        Use mean non-padding embeddings for count prediction.

        Parameters:
        -----------
        src_input_id: `torch.Tensor`
            Source token input.
        tgt_input_id_dict: `dict`
            Dictionary of target token inputs from different time steps.
        max_len: `int`
            Maximum length of the generated sequence.
        can_remask_prev_masked: `bool`
            Whether to remask previously masked tokens.
        topk_filter_thres: `float`
            Top-k filter threshold based on the logits.
        temperature: `float`
            Temperature to increase or decrease the randomness of the predictions.
        iterations: `int`
            Number of iterations until all tokens are predicted.
        mask_scheduler: `str`
            Mask scheduler function.
            Options: ['uniform', 'pow', 'cosine', 'log', 'exp']

        Returns:
        --------
        count_outputs: `dict`
            Output dictionary containing the following keys:
            - 'count_output_t{t}': Count prediction for time step t.
            - 'cls_embedding_t{t}': CLS token embeddings for time step t.

        '''
        # print('tgt_input_id', tgt_input_id[:5,:10])
        # use max shape instead of genes you like to generate
        # pad_tensor = torch.ones_like(tgt_input_id)

        # if pad_tensor.shape[1] > max_len:
        #     # set the rest of the tokens to zero
        #     pad_tensor[:, max_len:] = 0
        tgt_pad = generate_padding(src_input_id)
        # create ids and scores matrix for each batch
        ids = torch.full_like(tgt_input_id, self.mask_token, dtype=torch.long)
        # print(tgt_input_id[:5,:10])
        # add task token to the ids
        ids[:, 0] = tgt_input_id[:, 0]
        scores = torch.zeros_like(tgt_input_id, dtype=torch.float)
        scores = scores.masked_fill(tgt_pad, -torch.finfo().max)
        ids = ids.masked_fill(tgt_pad, 0)
        # scores[:, 0] = 1.0
        pred_ids = self.generate_sequence(
            ids=ids,
            tgt_pad=tgt_pad,
            src_input_id=src_input_id,
            mask_scheduler=mask_scheduler,
            can_remask_prev_masked=can_remask_prev_masked,
            topk_filter_thres=topk_filter_thres,
            starting_temperature=temperature,
            iterations=iterations,
            scores=scores,
        )
        # print('pred_ids:')
        # print(pred_ids[:5,:10])
        outputs = self.transformer(
            src_input_id=src_input_id,
            tgt_input_id=pred_ids,
            # generate_pad=tgt_pad,
            apply_attn_mask=self.apply_attn_mask,
        )
        return outputs, pred_ids

    @torch.no_grad()
    def generate_sequence(
        self,
        ids: torch.Tensor,
        tgt_pad: torch.Tensor,
        src_input_id: torch.Tensor,
        mask_scheduler: str,
        scores: torch.Tensor,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        starting_temperature: float = 2.0,
        iterations: int = 18,
    ):
        '''
        Description:
        ------------
        Generate sequences for the target tokens
        adopted from MaskGIT using the pretrained model.

        Parameters:
        -----------
        generate_id_dict: `dict`
            Dictionary of target token inputs for generation.
        generate_pad_dict: `dict`
            Dictionary of target padding masks for generation.
        src_input_id: `torch.Tensor`
            Source token input.
        demask_fn: `nn.Module`
            Pretrained model for demasking.
        mask_scheduler: `str`
            Mask scheduler function.
            Options: ['uniform', 'pow', 'cosine', 'log', 'exp']
        scores: `torch.Tensor`
            Probability scores for the tokens.
        can_remask_prev_masked: `bool`
            Whether to remask previously masked tokens.
        topk_filter_thres: `float`
            Top-k filter threshold based on the logits.
        starting_temperature: `float`
            Temperature to increase or decrease the randomness of the predictions.
        iterations: `int`
            Number of iterations until all tokens are predicted.

        Returns:
        --------
        outputs: `dict`
            Output dictionary containing the following keys:
            - 'dec_logits': Decoder logits.
            - 'labels': True labels for masked tokens.
            - 'selected_time_step': Selected time step.
            - 'dec_embedding': Decoder embeddings.
            - 'mean_embedding': Mean embeddings for non-padding tokens.
        ids: `torch.Tensor`
            Generated target token inputs.
        '''
        max_neg_value = -torch.finfo().max
        task_token = ids[:, 0]
        for iteration, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, iterations),
                reversed(range(iterations)),
            ),
            total=iterations,
        ):
            # mask scheduler function, gamma
            rand_mask_prob = noise_schedule(
                ratio=iteration,
                total_tokens=ids.shape[1],
                method=mask_scheduler,
            )
            ids_to_keep = torch.zeros_like(ids, dtype=torch.long)
            batch_size, seq_len = scores.shape
            unpadded = (scores != max_neg_value).sum(dim=1)
            num_token_masked = (unpadded.float() * rand_mask_prob).long()
            mask = torch.zeros_like(scores, dtype=torch.bool)
            masked_indices = torch.topk(scores, num_token_masked.max(), dim=-1).indices
            mask = mask.scatter(1, masked_indices, True)
            # Mask the top `num_tokens_to_mask` positions for each sample
            for i in range(batch_size):
                mask[i, masked_indices[i, : num_token_masked[i]]] = True
            ids = ids.masked_fill(mask, self.mask_token)
            # print('scores',scores[:5,:])
            # print('ids',ids[:5,:])
            # keep indices which are not masked
            ids[:, 0] = task_token
            ids_to_keep = torch.where(
                mask,
                torch.tensor(0, dtype=ids.dtype, device=ids.device),
                ids,
            )
            # demask tokens
            outputs = self.transformer(
                src_input_id=src_input_id,
                apply_attn_mask=False,
                generate_id=ids,
                generate_pad=tgt_pad,
            )
            logits = outputs['dec_logits']
            # exclude cls token
            ids_ = ids[:, 1:]
            scores_ = scores[:, 1:]
            ids_to_keep_ = ids_to_keep[:, 1:]
            # avoid predicting already predicted tokens
            # thus set the logits to max_neg_value
            unique_ids_per_sample = [torch.unique(ids) for ids in ids_to_keep_]
            # Create a mask for the logits to set specific positions to max_neg_value
            logits_mask = torch.zeros_like(logits).bool()

            for i, unique_ids in enumerate(unique_ids_per_sample):
                logits_mask[i, :, unique_ids] = True
            logits[logits_mask] = max_neg_value

            filtered_logits = top_k(logits, topk_filter_thres)
            temperature = starting_temperature * (
                steps_until_x0 / iteration
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = ids_ == self.mask_token
            ids_ = torch.where(is_mask, pred_ids, ids_)

            probs_without_temperature = logits.softmax(dim=-1)
            scores_ = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores_ = rearrange(scores_, '... 1 -> ...')

            if not can_remask_prev_masked:
                scores_ = scores_.masked_fill(~is_mask, max_neg_value)
            # add cls token to the ids and update scores and ids
            scores[:, 1:] = scores_
            ids[:, 1:] = ids_
        return ids

    def training_step(self, batch, *args, **kwargs):
        # logits, labels, count_output, count_dropout = self.forward(batch)
        outputs = self.forward(batch)
        categories = batch['disease']
        num_classes = len(set(categories))

        dec_logits = outputs['dec_logits']
        labels = outputs['labels']
        expert_logits_list = outputs['expert_logits_list']

        perp = self.perplexity(dec_logits, labels)
        dec_logits = dec_logits.contiguous().view(-1, dec_logits.size(-1))
        labels = labels.contiguous().view(-1)

        masking_loss = self.masking_loss(dec_logits, labels)

        if expert_logits_list is not None:
            # Create a mapping dictionary
            category_to_int = {
                category: idx for idx, category in enumerate(set(categories))
            }
            # Convert the category list to a tensor of indices
            category_indices = torch.tensor(
                [category_to_int[category] for category in categories],
                device=expert_logits_list[0].device,
            )
            # Efficient one-hot encoding using scatter_
            one_hot_labels = torch.zeros(
                len(categories), num_classes, device=expert_logits_list[0].device
            )
            one_hot_labels.scatter_(1, category_indices.unsqueeze(1), 1)

            # Calculate the BCE loss for each class
            expert_loss = [
                self.bce_loss(expert_logits_list[i].squeeze(-1), one_hot_labels[:, i])
                for i in range(num_classes)
            ]
            moe_loss = sum(expert_loss)

            self.log(
                'train/moe_loss',
                moe_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['src_input_ids'].shape[0],
                rank_zero_only=True,
                sync_dist=True,
            )
            total_loss = (1 - self.alpha) * masking_loss + self.alpha * moe_loss
        else:
            total_loss = masking_loss
        # # Convert class_column to one-hot encoded target matrix
        # target = torch.zeros(len(class_column), num_classes)
        # target[torch.arange(len(class_column)), class_column] = 1

        self.log(
            'train/masking_loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['src_input_ids'].shape[0],
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
            batch_size=batch['src_input_ids'].shape[0],
            rank_zero_only=True,
            sync_dist=True,
        )

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        dec_logits = outputs['dec_logits']
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
            batch_size=batch['src_input_ids'].shape[0],
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
            batch_size=batch['src_input_ids'].shape[0],
            rank_zero_only=True,
            sync_dist=True,
        )
        return masking_loss

    def test_step(self, batch, *args, **kwargs):
        if self.return_embeddings:
            tgt_ids = batch['tgt_input_ids']
            if self.generate:
                outputs, pred_ids = self.iterative_generate(
                    src_input_id=batch['src_input_ids'],
                    tgt_input_id=tgt_ids,
                    max_len=self.max_seq_length,
                    mask_scheduler=self.mask_scheduler,
                    can_remask_prev_masked=False,
                    topk_filter_thres=0.90,
                    temperature=self.temperature,
                    iterations=self.iterations,
                )
                pred_ids = pred_ids.cpu().numpy()
                tgt_ids = tgt_ids.cpu().numpy()
                # exclude task token and padding token
                # exclude padding token
                pred_ids = pred_ids[:, 1:]
                pred_ids = pred_ids[pred_ids != 0].tolist()
                tgt_ids = tgt_ids[:, 1:]
                tgt_ids = tgt_ids[tgt_ids != 0].tolist()

                pred_genes = [
                    str(self.tokenid_to_genename_dict.get(idx, idx)) for idx in pred_ids
                ]
                true_genes = [
                    str(self.tokenid_to_genename_dict.get(idx, idx)) for idx in tgt_ids
                ]
                pred_genes = ' '.join(pred_genes)
                true_genes = ' '.join(true_genes)
                # rouge score
                rouge_score = self.rouge(pred_genes, true_genes)

                self.log(
                    'test/rouge_f1',
                    rouge_score['rouge1_fmeasure'],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    rank_zero_only=True,
                    sync_dist=True,
                    batch_size=batch['src_input_ids'].shape[0],
                )
                self.log(
                    'test/rouge_precision',
                    rouge_score['rouge1_precision'],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    rank_zero_only=True,
                    sync_dist=True,
                    batch_size=batch['src_input_ids'].shape[0],
                )
                self.log(
                    'test/rouge_recall',
                    rouge_score['rouge1_recall'],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    rank_zero_only=True,
                    sync_dist=True,
                    batch_size=batch['src_input_ids'].shape[0],
                )

            else:
                outputs = self.forward(batch)
            token_ids = batch['tgt_input_ids']
            cell_ids = batch['tgt_cell_idx']
            # router_probs = outputs['router_probs']
            # print(f'router_probs: {router_probs[:15,:15]}')
            # print(router_probs.shape)
            cos_similarity, cls_embeddings, gene_embeddings = compute_cos_similarity(
                outputs=outputs,
                cls_mode='mean',
            )
            # define marker gene list to extract gene embeddings
            marker_genes = [
                'CSF2',
                'KDR',
                'NTM',
                'CCL2',
                'GPC5',
                'ROS1',
                'IL32',
                'TUBA1A',
                'MSLN',
                'DCBLD2',
                'FYN',
                'SLC34A2',
                'NR3C2',
                'ABLIM3',
                'WNT7A',
                'RRM2B',
                'MMP13',
            ]
            marker_cos_similarity, marker_genes_dict = return_cos_similarity(
                marker_genes=marker_genes,
                cos_similarity=cos_similarity,
                gene_embeddings=gene_embeddings,
                mapping_dict=self.tokenid_to_genename_dict,
                token_ids=token_ids,
            )
            marker_gene_embeddings = return_gene_embeddings(
                marker_genes=marker_genes,
                gene_embeddings=gene_embeddings,
                mapping_dict=self.tokenid_to_genename_dict,
                token_ids=token_ids,
            )
            self.marker_genes = marker_genes_dict
            self.test_dict['true_counts'].append(batch['tgt_counts'].detach().cpu())
            self.test_dict['cls_embeddings'].append(
                outputs['cls_embedding'].detach().cpu()
            )
            self.test_dict['mean_embeddings'].append(
                outputs['mean_embedding'].detach().cpu()
            )
            self.test_dict['cosine_similarities'].append(
                marker_cos_similarity.detach().cpu()
            )
            if batch['combined_batch'] is not None:
                self.test_dict['batch'].append(batch['combined_batch'].detach().cpu())
            self.test_dict['cell_idx'].append(cell_ids)
            self.test_dict['gene_embeddings'].append(
                marker_gene_embeddings.detach().cpu()
            )
            for var in self.var_list:
                self.test_dict[var].append(batch[var])

    def on_test_epoch_end(self):
        if self.return_embeddings:
            print('Start saving embeddings -------------------')
            cls_embeddings = torch.cat(self.test_dict['cls_embeddings'])
            true_counts = torch.cat(self.test_dict['true_counts'])
            cosine_similarities = torch.cat(self.test_dict['cosine_similarities'])
            var_dict = {}
            for var in self.var_list:
                var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)
            if 'combined_batch' in self.test_dict.keys():
                batch = torch.cat(self.test_dict['batch'])
                test_obs['batch'] = np.array(batch)
            cell_ids = np.concatenate(self.test_dict['cell_idx'])
            gene_embeddings = torch.cat(self.test_dict['gene_embeddings'])
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
                f'{self.output_dir}/{self.date}_'
                f'cls_embeddings_{self.moe_type}_generate_{self.generate}.h5ad'
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
        temperature: float = 2.0,
        iterations: int = 18,
        n_samples: int = 1,
        output_dir: str = './T_perturb/T_perturb/plt/res/eb/',
        mask_scheduler: Optional[str] = 'cosine',
        encoder_type: Literal[
            'GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'
        ] = 'GF_frozen',
        moe_type: Literal['moe_attention', 'none', 'moe_ffn'] = 'none',
        seed: int = 42,
        n_task_conditions: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        # Load pretrained masking transformer
        pretrained_model = CellGen(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            encoder_type=encoder_type,
            moe_type=moe_type,
            n_task_conditions=n_task_conditions,
        )
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
        self.time_steps = time_steps
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
        self.encoder_type = encoder_type
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
        self.val_pred_counts_list = []
        self.val_tgt_cell_type_list = []
        self.val_tgt_cell_population_list = []
        self.val_tgt_donor_list = []

    def test_step(self, batch, *args, **kwargs):
        tgt_input_id_dict = {}
        for i in range(1, self.n_task_conditions + 1):
            tgt_input_id_dict[f'tgt_input_id_t{i}'] = batch[f'tgt_input_ids_t{i}']

        if self.generate:
            outputs = self.decoder.generate(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                max_len=self.max_seq_length,
                mask_scheduler=self.mask_scheduler,
                can_remask_prev_masked=False,
                topk_filter_thres=0.9,
                temperature=self.temperature,
                iterations=self.iterations,
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
                f'{self.time_steps}__{self.encoder_type}_{self.seed}_'
                f'{self.loss_mode}_{self.n_samples}.h5ad'
            )
            emd = evaluate_emd(true_adata, pred_adata)
            self.log(
                'test/emd',
                emd['emd'].mean(),
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            print('---anndata generation completed')
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
            pred_adata.write_h5ad(f'{self.output_dir}/{self.date}_pred_adata.h5ad')
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
            mse = self.mse(pred_counts, true_counts)

            metrics = pd.DataFrame(
                {
                    'pearson': [mean_pearson.cpu().detach().numpy()],
                    'pearson_delta': [mean_pearson_delta.cpu().detach().numpy()],
                    'mse': [mse.cpu().detach().numpy()],
                }
            )
            metrics.to_csv(f'{self.output_dir}/test_metrics.csv')
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
