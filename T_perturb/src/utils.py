import argparse
import math
import os
import pickle
import re
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
)

import anndata as ad
import geneformer.perturber_utils as pu
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tqdm
from datasets import DatasetDict, load_from_disk
from geneformer import EmbExtractor
from geneformer.emb_extractor import get_embs, label_cell_embs
from scipy.sparse import csr_matrix
from torch.nn.functional import cosine_similarity
from torch.optim import Optimizer
from torch.utils.data import Subset
from torchmetrics import PearsonCorrCoef


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        initial_lr: float,
        end_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        if current_step < self.warmup_steps:
            # Linear warmup phase: increase from initial_lr to end_lr
            warmup_lr = [
                self.initial_lr
                + (self.end_lr - self.initial_lr) * (current_step / self.warmup_steps)
                for _ in self.base_lrs
            ]
            return warmup_lr
        else:
            # After warmup, maintain the end_lr (constant LR)
            return [self.end_lr for _ in self.base_lrs]


def read_dataset_files(directory, file_type):
    dataset_dict = {}
    for filename in os.listdir(directory):
        print(f'Loading {filename}...')
        if filename.endswith(f'.{file_type}'):
            filename_ = os.path.join(directory, filename)
            if file_type == 'dataset':
                dataset_dict[f'tgt_{file_type}_t{filename[0]}'] = load_from_disk(
                    filename_
                )  # Removing the '.dataset' extension from the key
            elif file_type == 'h5ad':
                dataset_dict[f'tgt_{file_type}_t{filename[0]}'] = sc.read_h5ad(
                    filename_
                )
    return dataset_dict


def map_ensembl_to_genename(
    adata: ad.AnnData,
    mapping_path: str,
) -> ad.AnnData:
    """
    Description:
    ------------
    This function maps ensembl ids to gene names.
    """
    # read in .csv file to map ensembl ids to gene names
    mapping_df = pd.read_csv(mapping_path)
    # rename column gene_ids to ensembl_id
    mapping_df = mapping_df.rename(columns={'gene_ids': 'ensembl_id'})
    # left join adata.var with mapping_df to add ensembl ids to adata.var
    adata.var['gene_name'] = adata.var_names
    adata.var = adata.var.merge(
        mapping_df[['index', 'ensembl_id']],
        left_on='gene_name',
        right_on='index',
        how='left',
    )
    # create ensembl_id column and drop index and ensembl_id columns
    adata.var_names = adata.var['ensembl_id']
    adata.var = adata.var.drop(columns=['index', 'ensembl_id'])
    return adata


def condition_for_count_loss(
    condition_keys: str | list[str] | None,
    conditions: dict | None,
    conditions_combined: list | None,
    tgt_adata_tmp: ad.AnnData,
):
    '''
    Description:
    ------------
    This function encodes conditions for count loss.
    Parameters:
    -----------
    condition_keys: `str`
        Key to encode conditions.
    conditions: `dict`
        Dictionary of conditions.
    conditions_combined: `list`
        List of combined conditions.
    tgt_adata_tmp: `~anndata.AnnData`
        target adata object.
    Returns:
    --------
    conditions: `torch.tensor`
        Tensor of encoded conditions.
    condition_encodings: `dict`
        Dictionary of condition encodings.
    conditions_combined: `torch.tensor`
        Tensor of encoded combined conditions.
    conditions_: `dict`
        Dictionary of conditions.
    condition_keys_: `list`
        List of condition keys.
    conditions_combined_: `list`
        List of combined conditions.
    '''
    if condition_keys is None:
        condition_keys = 'tmp_batch'
        # create a mock vector if there are no batch effect
        tgt_adata_tmp.obs[condition_keys] = 1
    if isinstance(condition_keys, str):
        condition_keys_ = [condition_keys]
    else:
        condition_keys_ = condition_keys
    if conditions is None:
        if condition_keys is not None:
            conditions_ = {}
            for cond in condition_keys_:
                conditions_[cond] = tgt_adata_tmp.obs[cond].unique().tolist()
        else:
            conditions_ = {}
    else:
        conditions_ = conditions
    if conditions_combined is None:
        if len(condition_keys_) > 1:
            tgt_adata_tmp.obs['conditions_combined'] = tgt_adata_tmp.obs[
                condition_keys
            ].apply(lambda x: '_'.join(x), axis=1)
        else:
            tgt_adata_tmp.obs['conditions_combined'] = tgt_adata_tmp.obs[condition_keys]
        conditions_combined_ = (
            tgt_adata_tmp.obs['conditions_combined'].unique().tolist()
        )
    else:
        conditions_combined_ = conditions_combined
    condition_encodings: Dict[str, Dict[str, int]] = {
        cond: {k: v for k, v in zip(conditions_[cond], range(len(conditions_[cond])))}
        for cond in conditions_.keys()
    }
    conditions_combined_encodings = {
        k: v for k, v in zip(conditions_combined_, range(len(conditions_combined_)))
    }
    if (condition_encodings is not None) and (condition_keys_ is not None):
        conditions_tmp = [
            label_encoder(
                tgt_adata_tmp,
                encoder=condition_encodings[condition_keys_[i]],
                condition_key=condition_keys_[i],
            )
            for i in range(len(condition_encodings))
        ]
        conditions = torch.tensor(conditions_tmp, dtype=torch.long).T
        conditions_combined = label_encoder(
            tgt_adata_tmp,
            encoder=conditions_combined_encodings,
            condition_key='conditions_combined',
        )
        conditions_combined = torch.tensor(conditions_combined, dtype=torch.long)
    return (
        conditions,
        condition_encodings,
        conditions_combined,
        conditions_,
        condition_keys_,
        conditions_combined_,
    )


def get_idx_for_filtering(
    dataset: DatasetDict,
    condition: list[str],
    variable: str,
):
    return [i for i, val in enumerate(dataset[variable]) if val in condition]


def concat_cond_tokens(
    batch: dict,
    time_step: int,
    condition_dict: dict[str, dict] | None = None,
):
    tgt_input_ids = batch[f'tgt_input_ids_t{time_step}']
    device = tgt_input_ids.device
    batch_size = tgt_input_ids.shape[0]
    if condition_dict is not None:
        cond_ids = torch.zeros(
            (batch_size, len(condition_dict)), dtype=torch.long, device=device
        )
        for j, condition in enumerate(condition_dict.keys()):
            condition_tokens = [
                condition_dict[condition][id]
                for id in batch[f'{condition}_t{time_step}']
            ]
            # j+1 to skip time token
            cond_ids[:, j] = torch.tensor(
                condition_tokens, dtype=torch.long, device=device
            )

    return cond_ids


def compute_rouge_score(
    rouge,
    pred_ids: np.ndarray,
    tgt_ids: np.ndarray,
    rouge_len_list: list[int],
    max_seq_length: int,
    test_dict: dict[str, list],
    cond_length: int | None = None,
) -> dict[str, list[float]]:
    # Convert predictions and targets to object type for string replacement
    pred_ids = pred_ids.astype(object)
    tgt_ids = tgt_ids.astype(object)

    # # Exclude task token (assumed to be the first token)
    pred_ids = pred_ids[:, cond_length:]

    # Define special tokens to remove (e.g., 0, 1, 2, 3)
    special_tokens = np.array([0, 1, 2, 3])

    pred_ids[np.isin(pred_ids, special_tokens)] = ''
    tgt_ids[np.isin(tgt_ids, special_tokens)] = ''

    # Convert all integers to strings
    pred_ids_ = pred_ids.astype(str)
    tgt_ids_ = tgt_ids.astype(str)

    for seq_len in rouge_len_list:
        # Truncate sequences based on seq_len
        if max_seq_length > seq_len:
            pred_genes_short = pred_ids_[:, :seq_len]
            true_genes_short = tgt_ids_[:, :seq_len]
        else:
            pred_genes_short = pred_ids_
            true_genes_short = tgt_ids_

        # Convert each row of token IDs to a string
        pred_ids_str = np.apply_along_axis(
            lambda row: ' '.join(row), axis=1, arr=pred_genes_short
        )
        tgt_ids_str = np.apply_along_axis(
            lambda row: ' '.join(row), axis=1, arr=true_genes_short
        )

        # Remove extra spaces
        pred_ids_str = np.array([' '.join(s.split()) for s in pred_ids_str])
        pred_ids_str = pred_ids_str.tolist()
        tgt_ids_str = np.array([' '.join(s.split()) for s in tgt_ids_str])
        tgt_ids_str = tgt_ids_str.tolist()

        # Compute ROUGE score for the entire batch at once
        rouge_result = rouge.compute(
            predictions=pred_ids_str,
            references=tgt_ids_str,
            rouge_types=['rouge1'],
            use_aggregator=False,
        )
        # Extract the per-sample ROUGE scores
        per_sample_scores = rouge_result['rouge1']
        # Store the results in the test_dict
        test_dict[f'rouge1_{seq_len}'].append(per_sample_scores)

    return test_dict


def compute_cos_similarity(
    gene_embeddings: torch.tensor,
    mean_embedding: torch.tensor,
):
    """
    Description:
    ------------
    This function computes cosine similarity between cls and gene embeddings.
    Parameters:
    -----------
    gene_embeddings: `torch.tensor`
        Tensor of gene embeddings.
    mean_embedding: `torch.tensor`
        Tensor of mean aggregated cls embeddings.
    Returns:
    --------
    cos_similarity: `torch.tensor`
        Tensor of cosine similarity between cls and gene embeddings.
    """
    cos_similarity = []
    # iterate over batch/cell dimension
    for i in range(gene_embeddings.shape[0]):
        # gene level cosine similarity
        cos_similarity_ = cosine_similarity(
            mean_embedding[i],
            gene_embeddings[i, :, :],
            dim=1,
        )
        cos_similarity.append(cos_similarity_)
    cos_similarity = torch.stack(cos_similarity)

    return cos_similarity


def map_results_to_genes(
    res: torch.tensor,
    mapping_dict: Dict,
    token_ids: torch.tensor,
    marker_genes: List[str] | None = None,
) -> torch.tensor:
    """
    Description:
    ------------
    This function returns cosine similarity for marker genes.
    Parameters:
    -----------
    marker_genes: `List[str]`
        List of marker genes.
    res: `torch.tensor`
        Tensor containing results ordered by token ids.
    mapping_dict: `Dict`
        Dictionary mapping gene names to token ids.
    Returns:
    --------
    res_: `torch.tensor`
        Tensor reordered by gene names.
    marker_genes_dict: `Dict`
        Dictionary of marker genes -> {gene: index}
    """
    # filter for marker genes and swap key value
    if marker_genes is not None:
        marker_genes_ids = {v: k for v, k in mapping_dict.items() if v in marker_genes}
    else:
        # exclude special tokens from marker genes
        special_tokens = ['<cls>', '<mask>', '<pad>', '<eos>']
        marker_genes_ids = {
            v: k for v, k in mapping_dict.items() if v not in special_tokens
        }
    res_ = torch.zeros(
        res.shape[0],
        len(marker_genes_ids.keys()),
        device=res.device,
    )
    marker_genes_dict = {}
    for i, gene in enumerate(marker_genes_ids.keys()):
        # extract cosine similarity for marker genes
        # ---------------------
        cond_embs_to_fill = (token_ids == marker_genes_ids[gene]).sum(1) > 0
        cond_select_markers = torch.where(token_ids == marker_genes_ids[gene])
        res_[cond_embs_to_fill, i] = res[cond_select_markers[0], cond_select_markers[1]]
        marker_genes_dict[gene] = i
    return res_, marker_genes_dict


def return_attn_weights(
    outputs: dict,
    src_mapping_dict: Dict,
    tgt_mapping_dict: Dict,
    time_step: int,
    token_ids: torch.tensor,
    pad_token_id: List[str],
    marker_genes: Optional[List[str]] = None,
    context_token_ids: Optional[torch.tensor] = None,
):
    """
    Description:
    ------------
    This function maps token ids to gene names for attention weights.

    Parameters:
    -----------
    outputs: `dict`
        Dictionary containing outputs from the model.
    src_mapping_dict: `Dict`
        Dictionary mapping gene names to token ids.
    time_step: `int`
        Time step to compute cosine similarity.
    token_ids: `torch.tensor`
        Tensor of token ids from target tensor.
    marker_genes: `List[str]`
        List of marker genes.
    pad_token_id: int
        exclude pad token from attention weights results.

    Returns:
    --------
    attn_weights_res: `torch.tensor`
    """

    # filter for marker genes and swap key value
    # if marker_genes is not None:
    #     marker_genes_ids = {
    #         v: k for v, k in tgt_mapping_dict.items() if v in marker_genes
    #     }
    # else:

    # map self attention weights
    pad_token_id = torch.tensor(pad_token_id, device=token_ids.device)
    self_attn_weights = outputs[time_step]['self_attn_weights']
    self_attn_weights = _map_attn_weights(
        attn_weights=self_attn_weights,
        tgt_mapping_dict=tgt_mapping_dict,
        token_ids=token_ids,
        pad_token_id=pad_token_id,
    )
    # map cross attention weights
    cross_attn_weights = outputs[time_step]['cross_attn_weights']
    cross_attn_weights = _map_attn_weights(
        attn_weights=cross_attn_weights,
        src_mapping_dict=src_mapping_dict,
        tgt_mapping_dict=tgt_mapping_dict,
        token_ids=token_ids,
        pad_token_id=pad_token_id,
        context_token_ids=context_token_ids,
    )

    return self_attn_weights, cross_attn_weights


def _map_attn_weights(
    attn_weights: torch.tensor,
    tgt_mapping_dict: Dict,
    token_ids: torch.tensor,
    pad_token_id: torch.tensor,
    context_token_ids: Optional[torch.tensor] = None,
    src_mapping_dict: Optional[Dict] = None,
):
    batch_size = attn_weights.shape[0]
    # exclude cls token from gene embeddings
    tgt_n_genes = len(tgt_mapping_dict.keys())
    if (src_mapping_dict is not None) and (context_token_ids is not None):
        # remap token ids to new token ids for multidimensional tensor
        src_token_ids = context_token_ids[0]
        src_token_ids = torch.tensor(
            [src_mapping_dict[int(token)] for token in src_token_ids.flatten()],
            device=src_token_ids.device,
        ).reshape(src_token_ids.shape)
        context_token_ids[0] = src_token_ids
        context_n_genes = len(src_mapping_dict.keys()) + len(
            tgt_mapping_dict.keys()
        ) * (
            len(context_token_ids) - 1
        )  # exlude src
    else:
        context_n_genes = tgt_n_genes

    attn_weights_res = torch.zeros(
        size=(batch_size, tgt_n_genes, context_n_genes),
        device=attn_weights.device,
        dtype=attn_weights.dtype,
    )

    for i in range(batch_size):
        token_idx = token_ids[i]  # Shape (seq_len)
        attn_weights_ = attn_weights[i]  # Shape (seq_len, seq_len)
        # sort token indices
        if context_token_ids is None:
            sorted_tokens, sorted_indices = torch.sort(token_idx)
            # remove special tokens from sorted tokens and indices
            sorted_tokens_ = sorted_tokens[~torch.isin(sorted_tokens, pad_token_id)]
            sorted_indices_ = sorted_indices[~torch.isin(sorted_tokens, pad_token_id)]
            sorted_attn_matrix = attn_weights_[sorted_indices_][:, sorted_indices_]
            attn_weights_res = attn_weights_res.clone()
            tgt_indices_0 = sorted_tokens_.unsqueeze(1).expand(
                -1, sorted_attn_matrix.shape[1]
            )
            tgt_indices_1 = sorted_tokens_.unsqueeze(0).expand(
                sorted_attn_matrix.shape[0], -1
            )
            attn_weights_res[i, tgt_indices_0, tgt_indices_1] = sorted_attn_matrix
        else:
            sorted_tgt_tokens, sorted_tgt_indices = torch.sort(token_idx)
            sorted_tgt_tokens_ = sorted_tgt_tokens[
                ~torch.isin(sorted_tgt_tokens, pad_token_id)
            ]
            sorted_tgt_indices_ = sorted_tgt_indices[
                ~torch.isin(sorted_tgt_tokens, pad_token_id)
            ]
            # adjust for second cross attention provided as context
            context_seq_len = 0
            # adjust to fill in at correct location in attn_weights_res
            context_vocab_size = 0
            for context_token in context_token_ids:
                context_token_idx = context_token[i]
                sorted_context_tokens, sorted_context_indices = torch.sort(
                    context_token_idx
                )
                sorted_context_tokens_ = sorted_context_tokens[
                    ~torch.isin(sorted_context_tokens, pad_token_id)
                ]
                sorted_context_indices_ = sorted_context_indices[
                    ~torch.isin(sorted_context_tokens, pad_token_id)
                ]
                sorted_attn_matrix = attn_weights_[sorted_tgt_indices_][
                    :, context_seq_len + sorted_context_indices_
                ]
                # adjust context token indices by context_seq_len

                tgt_indices = sorted_tgt_tokens_.unsqueeze(1).expand(
                    -1, sorted_attn_matrix.shape[1]
                )
                adjusted_context_tokens = sorted_context_tokens_ + context_vocab_size
                context_indices = adjusted_context_tokens.unsqueeze(0).expand(
                    sorted_attn_matrix.shape[0], -1
                )
                attn_weights_res[
                    i,
                    tgt_indices,
                    context_indices,
                ] = sorted_attn_matrix
                context_seq_len += len(context_token_idx)
                if (context_vocab_size == 0) and (src_mapping_dict is not None):
                    # add src mapping dict fist
                    context_vocab_size += len(src_mapping_dict.keys())
                else:
                    context_vocab_size += len(tgt_mapping_dict.keys())
        return attn_weights_res


def aggregate_attn_weights(
    attn_weights: torch.tensor,
    tgt_gene_names: List[str],
    output_dir: str,
    file_name: str,
    src_gene_names: Optional[List[str]] = None,
):
    """
    attn_weights: `torch.tensor`
        Tensor of attention weights.
    output_dir: `str`
        Directory to save files in
    file_name: `str`
        Filename for output file
    """
    # take the mean of attention weights
    attn_weights_mean = attn_weights.mean(0).T
    # create a dataframe of attention weights

    if src_gene_names is not None:
        tgt_len = attn_weights_mean.shape[0] - len(src_gene_names)
        repeat_factor = tgt_len // len(tgt_gene_names)
        context_gene_names = src_gene_names + tgt_gene_names * repeat_factor
    else:
        context_gene_names = tgt_gene_names

    attn_weights_df = pd.DataFrame(
        attn_weights_mean.cpu().numpy(),
        index=context_gene_names,
        columns=tgt_gene_names,
    )
    # remove all zero rows and columns
    attn_weights_df = attn_weights_df.loc[
        (attn_weights_df != 0).any(axis=1), (attn_weights_df != 0).any(axis=0)
    ]
    print(f'Saving attention weights to {output_dir}/{file_name} ...')
    attn_weights_df.to_csv(os.path.join(output_dir, f'{file_name}.csv'))


def exclude_special_tokens(
    mapping_dict: dict,
    marker_genes: list[str] | None = None,
):
    if marker_genes is not None:
        marker_genes_ids = {v: k for v, k in mapping_dict.items() if v in marker_genes}
    else:
        special_tokens = ['<cls>', '<mask>', '<pad>', '<eos>']
        marker_genes_ids = {
            v: k for v, k in mapping_dict.items() if v not in special_tokens
        }
        # exclude all tokens starting with cls
        marker_genes_ids = {
            v: k for v, k in marker_genes_ids.items() if not v.startswith('cls')
        }
    return marker_genes_ids


def mask_duplicates_across_batches(
    aggregate_cell_idx: np.array,
    cell_idx: np.array,
):
    if len(aggregate_cell_idx) == 0:
        return (
            np.ones(len(cell_idx), dtype=bool),
            cell_idx,
        )
    else:
        intersect = np.intersect1d(cell_idx, aggregate_cell_idx)
        if len(intersect) > 0:
            mask = np.isin(cell_idx, intersect)
            inv_mask = ~mask
            cell_idx = cell_idx[inv_mask]
            # invert mask to exclude duplicates
            return inv_mask, cell_idx
        else:
            return (
                np.ones(len(cell_idx), dtype=bool),
                cell_idx,
            )


def mask_duplicates_within_batches(
    cell_idx: np.array,
):
    if len(set(cell_idx)) < len(cell_idx):
        _, first_indices = np.unique(cell_idx, return_index=True)
        # Create a mask where the first occurrence of each element is True
        mask = np.zeros(len(cell_idx), dtype=bool)
        mask[first_indices] = True

        return mask, cell_idx[mask]
    else:
        return (np.ones(len(cell_idx), dtype=bool), cell_idx)


def return_gene_embeddings(
    gene_embeddings: torch.tensor,
    mapping_dict: Dict,
    token_ids: torch.tensor,
    marker_genes: Optional[List[str]] = None,
) -> torch.tensor:
    """
    Description:
    ------------
    This function returns gene embeddings from a list of marker genes.
    # Parameters:
    -----------
    marker_genes: `List[str]`
        List of marker genes.
    gene_embeddings: `torch.tensor`
        Tensor of gene embeddings.
    mapping_dict: `Dict`
        Dictionary mapping gene names to token ids.
    token_ids: `torch.tensor`
        Tensor of token ids from target tensor.
    Returns:
    --------
    gene_embeddings_res: `torch.tensor`
    """
    marker_genes_ids = exclude_special_tokens(mapping_dict, marker_genes)
    # filter for marker genes and swap key value
    # marker_genes_ids = {v: k for v, k in mapping_dict.items() if v in marker_genes}
    gene_embeddings_res = torch.zeros(
        gene_embeddings.shape[0],
        len(marker_genes_ids.keys()),
        gene_embeddings.shape[2],
        device=gene_embeddings.device,
        dtype=gene_embeddings.dtype,
    )
    marker_genes_dict = {}
    for i, gene in enumerate(marker_genes_ids.keys()):
        # extract gene embeddings for marker genes
        # ---------------------
        cond_embs_to_fill = (token_ids == marker_genes_ids[gene]).sum(1) > 0
        cond_select_markers = torch.where(token_ids == marker_genes_ids[gene])
        gene_embeddings_res[cond_embs_to_fill, i, :] = gene_embeddings[
            cond_select_markers[0], cond_select_markers[1], :
        ]
        marker_genes_dict[gene] = i

    return gene_embeddings_res


def return_prediction_adata(
    test_dict: dict,
    obs_key: list,
    marker_genes: dict,
    output_dir: str,
    file_name: str,
    gene_names: list,
    sum_gene_embs: dict | None = None,
):
    """
    Description:
    ------------
    This function returns anndata object with predicted counts.
    Parameters:
    -----------
    test_dict: `dict`
        Dictionary containing test data.
    obs_key: `list`
        List of keys to include in adata.obs.
    marker_genes: `dict`
        List of marker genes.
    output_dir: `str`
        Directory to save files in
    file_name: `str`
        Filename for output file
    gene_names: `list`
        Name of var_names for adata.var.
    Returns:
    --------
    adata: `~anndata.AnnData` \n
        Annotated data matrix with predicted counts. \n
        - adata.X: `~numpy.ndarray`
            Array of predicted counts.
        - adata.obs: `~pandas.DataFrame`
            DataFrame of obs_key.
        - adata.var: `~pandas.DataFrame`
            DataFrame of gene names.
        - adata.obsm: `dict`
            'cls_embeddings': `~numpy.ndarray`
                Array of cls embeddings.
            'gene_embeddings': `~numpy.ndarray`
                Array of gene embeddings.
            'cosine_similarity': `~pandas.DataFrame`
                DataFrame of cosine similarities.
    """
    print('---Start saving embeddings')
    # adata.X
    true_counts = torch.cat(test_dict['true_counts'], dim=0).numpy()
    # adata.obsm
    cls_embeddings = torch.cat(test_dict['cls_embeddings'], dim=0).numpy()
    # adata.obs
    obs_dict = {obs: np.concatenate(test_dict[obs]) for obs in obs_key}
    test_obs = pd.DataFrame(obs_dict)
    # adata.var
    if gene_names is not None:
        test_var = pd.DataFrame(gene_names, columns=['gene_name'])
    # if len(test_dict['gene_embeddings']) > 0:
    #     gene_embeddings_dict = {}
    #     gene_embeddings = torch.cat(test_dict['gene_embeddings'], dim=0).numpy()
    #     gene_embeddings_dict['gene_embeddings'] = gene_embeddings

    #     # save as pkl file for downstream analysis
    #     with open(
    #         os.path.join(output_dir, f'{file_name}_gene_embeddings.pkl'), 'wb'
    #     ) as f:
    #         pickle.dump(gene_embeddings_dict, f)
    #     del gene_embeddings_dict

    adata = ad.AnnData(
        X=true_counts,
        obs=test_obs,
        var=test_var if gene_names is not None else None,
        obsm={
            'cls_embeddings': cls_embeddings,
        },
        uns={
            'marker_genes': marker_genes if marker_genes is not None else {},
        },
    )
    if sum_gene_embs is not None:
        for condition in sum_gene_embs.keys():
            adata.varm[condition] = sum_gene_embs[condition]
    if gene_names is not None:
        adata.var_names = adata.var['gene_name']
        adata.var = adata.var.drop(columns=['gene_name'])
    adata.write_h5ad(os.path.join(output_dir, f'{file_name}.h5ad'))
    if len(test_dict['cosine_similarities']) > 0:
        # save cosine similarity separately due to large size
        cos_similarity = torch.cat(test_dict['cosine_similarities'], dim=0).numpy()
        cos_similarity_df = pd.DataFrame(cos_similarity, columns=marker_genes.keys())
        # remove all non-expressed genes
        cos_similarity_df = cos_similarity_df.loc[:, cos_similarity_df.sum() != 0]
        # save as csv file for downstream analysis
        cos_similarity_df.index = adata.obs.index
        cos_similarity_df.to_csv(
            os.path.join(output_dir, f'{file_name}_cosine_similarity.csv')
        )
    # adata.obsm['cosine_similarity'] = cos_similarity_df
    print('End saving embeddings---')


def return_generation_adata(
    test_dict: dict,
    obs_key: list,
    output_dir: str,
    file_name: str,
):
    """
    Description:
    ------------
    This function returns anndata object with predicted counts.
    Parameters:
    -----------
    test_dict: `dict`
        Dictionary containing test data.
    obs_key: `list`
        List of keys to include in adata.obs.
    output_dir: `str`
        Directory to save files in
    file_name: `str`
        Filename for output file
    Returns:
    --------
    adata: `~anndata.AnnData` \n
        Annotated data matrix with predicted counts. \n
        - adata.X: `~numpy.ndarray`
            Array of predicted counts.
        - adata.obs: `~pandas.DataFrame`
            DataFrame of obs_key.
        - adata.var: `~pandas.DataFrame`
            DataFrame of gene names.
        - adata.obsm: `dict`
            'cls_embeddings': `~numpy.ndarray`
                Array of cls embeddings.
        - adata.layers: `~numpy.ndarray`
                Array of true counts.
    """
    print('---Generating anndata')
    # TODO: clean up no if and else needed
    # adata.X
    if 'pred_counts' in test_dict.keys():
        pred_counts = torch.cat(test_dict['pred_counts']).numpy()
    # adata.layers['counts']
    if 'true_counts' in test_dict.keys():
        true_counts = torch.cat(test_dict['true_counts']).numpy()
    # adata.obsm
    cls_embeddings = torch.cat(test_dict['cls_embeddings']).numpy()
    # adata.obs
    obs_dict = {obs: np.concatenate(test_dict[obs]) for obs in obs_key}
    test_obs = pd.DataFrame(obs_dict)
    # create adata
    if 'pred_counts' in test_dict.keys() and 'true_counts' in test_dict.keys():
        adata = ad.AnnData(
            X=pred_counts,
            obs=test_obs,
            obsm={'cls_embeddings': cls_embeddings},
            layers={'counts': true_counts},
        )
    else:
        adata = ad.AnnData(
            obs=test_obs,
            obsm={'cls_embeddings': cls_embeddings},
        )
    if 'true_embeddings' in test_dict.keys():
        true_embeddings = torch.cat(test_dict['true_embeddings']).numpy()
        adata.obsm['true_embeddings'] = true_embeddings
    adata.write_h5ad(os.path.join(output_dir, file_name))
    print('anndata generation completed---')
    return adata


def return_perturbation_adata(
    test_dict: dict,
    obs_key: list,
    output_dir: str,
    marker_genes: dict,
    file_name: str,
    mode: Literal['inference', 'generate'],
) -> ad.AnnData:
    """
    Description:
    ------------
    This function returns anndata object with predicted counts.
    Parameters:
    -----------
    test_dict: `dict`
        Dictionary containing test data.
    obs_key: `list`
        List of keys to include in adata.obs.
    output_dir: `str`
        Directory to save files in
    file_name: `str`
        Filename for output file
    mode: `Literal['inference', 'generation']`
        Mode of test_step.
    Returns:
    --------
    adata: `~anndata.AnnData` \n
        Annotated data matrix with predicted counts. \n
        - adata.X: `~numpy.ndarray`
            Array of perturbed counts.
        - adata.obs: `~pandas.DataFrame`
            DataFrame of obs_key.
        - adata.var: `~pandas.DataFrame`
            DataFrame of gene names.
        - adata.obsm: `dict`
            'cosine_similarity': `~pandas.DataFrame`
                DataFrame of cosine similarities between true and predicted embeddings.
            'rouge_scores': `~pandas.DataFrame`
                DataFrame of rouge scores between true and predicted sequences.
        - adata.layers: `~numpy.ndarray`
                Array of true counts.
    """
    print('---Generating anndata')
    # adata.obsm
    true_cls = torch.cat(test_dict['true_cls']).numpy()
    perturbed_cls = torch.cat(test_dict['perturbed_cls']).numpy()
    # adata.X
    pert_counts = None
    if 'pert_counts' in test_dict.keys():
        if len(test_dict['pert_counts']) > 0:
            pert_counts = torch.cat(test_dict['pert_counts']).numpy()
    # adata.layers['counts']
    true_counts = None
    if 'true_counts' in test_dict.keys():
        if len(test_dict['true_counts']) > 0:
            true_counts = torch.cat(test_dict['true_counts']).numpy()

    # cls_cos_similarity = torch.cat(test_dict['cls_cosine_similarity']).numpy()
    mean_cos_similarity = torch.cat(test_dict['mean_cosine_similarity']).numpy()
    if 'mean_cosine_similarity_l1' in test_dict.keys():
        mean_cos_similarity_l1 = torch.cat(
            test_dict['mean_cosine_similarity_l1']
        ).numpy()
    else:
        mean_cos_similarity_l1 = None
    if 'mean_cosine_similarity_lmid' in test_dict.keys():
        mean_cos_similarity_lmid = torch.cat(
            test_dict['mean_cosine_similarity_lmid']
        ).numpy()
    else:
        mean_cos_similarity_lmid = None
    # delta_probs = torch.cat(test_dict['delta_probs']).numpy()
    # wasserstein_distance = np.concatenate(test_dict['wasserstein_distance'])
    # adata.varm
    gene_cos_similarity = torch.cat(test_dict['gene_cosine_similarity'], dim=0).numpy()
    cos_similarity_df = pd.DataFrame(gene_cos_similarity, columns=marker_genes.keys())
    # cos_similarity_df = cos_similarity_df.T
    # cos_similarity_df.columns = cos_similarity_df.columns.astype(str)
    # delta_gene_probs = torch.cat(test_dict['delta_gene_probs'], dim=0).numpy()
    # delta_gene_probs_df = pd.DataFrame(
    #     delta_gene_probs, columns=marker_genes.keys()
    # )

    # create dataframe to store perturbation results
    obsm_dict = {
        'true_cls': true_cls,
        'perturbed_cls': perturbed_cls,
        # 'cls_cos_similarity': cls_cos_similarity,
        'mean_cos_similarity': mean_cos_similarity,
        'mean_cos_similarity_l1': mean_cos_similarity_l1,
        'mean_cos_similarity_lmid': mean_cos_similarity_lmid,
        # 'delta_probs': delta_probs,
    }
    cos_similarity_df_ = cos_similarity_df.T
    # convert columns to string
    cos_similarity_df_.columns = cos_similarity_df_.columns.astype(str)
    varm_dict = {
        'gene_cos_similarity': cos_similarity_df_,
    }

    #     # 'delta_gene_probs': delta_gene_probs_df.T,
    # }

    if mode == 'generate':
        rouge_dict = {
            key: np.concatenate(test_dict[key])
            for key in test_dict.keys()
            if key.startswith('rouge')
        }
        obsm_dict.update(rouge_dict)
    # adata.obs
    obs_dict = {obs: np.concatenate(test_dict[obs]) for obs in obs_key}
    test_obs = pd.DataFrame(obs_dict)
    # create adata
    adata = ad.AnnData(
        X=pert_counts,
        obs=test_obs,
        obsm=obsm_dict,
        varm=varm_dict,
        var=pd.DataFrame(
            index=cos_similarity_df.columns,
        ),
        layers={'counts': true_counts},
    )
    adata.write_h5ad(os.path.join(output_dir, file_name))
    print('anndata generation completed---')
    return adata


def scale_pca(adata):
    '''
    Description
    ------------
    This function returns scaled PCA on log-normalised counts
    to compute EMD and MMD.
    '''
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=5)
    # scale PCA
    coords = adata.obsm['X_pca']
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    adata.obsm['X_pca_scaled'] = coords
    return adata


def modify_ckpt_state_dict(
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


def pearson(
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    ctrl_counts: torch.Tensor = None,
) -> torch.Tensor:
    """
    Description:
    ------------
    This function computes the Pearson correlation coefficient for delta counts
    between control and perturbed conditions.
    Parameters:
    -----------
    pred_counts: `torch.Tensor`
        Tensor of predicted counts.
    true_counts: `torch.Tensor`
        Tensor of counts from perturbed condition.
    ctrl_counts: `torch.Tensor`
        Tensor of counts from control condition.
    Returns:
    --------
    mean_pearson: `torch.Tensor`
        Mean Pearson correlation coefficient.
    """
    if ctrl_counts is not None:
        pred_counts = pred_counts - ctrl_counts
        true_counts = true_counts - ctrl_counts
    num_outputs = true_counts.shape[0]
    pearson = PearsonCorrCoef(num_outputs=num_outputs).to(pred_counts.device)
    pred_counts_t = pred_counts.transpose(0, 1)
    true_counts_t = true_counts.transpose(0, 1)
    pearson_output = pearson(pred_counts_t, true_counts_t)
    mean_pearson = torch.mean(pearson_output)
    return mean_pearson


def subset_adata_dataset(
    src_adata: ad.AnnData,
    tgt_adata: ad.AnnData,
    src_dataset: DatasetDict,
    tgt_dataset: DatasetDict,
    num_cells: int,
    seed: int = 42,
):
    """
    Description:
    ------------
    This function ensures that all datasets have the same cell numbers.
    The cells are sampled randomly from the source and target datasets.
    Especially useful for code testing and debugging.

    Parameters:
    -----------
    src_adata: `~anndata.AnnData`
        Source annotated data matrix.
    tgt_adata: `~anndata.AnnData`
        Target annotated data matrix.
    src_dataset: `~datasets.DatasetDict`
        Source dataset.
    tgt_dataset: `~datasets.DatasetDict`
        Target dataset.
    num_cells: `int`
        Number of cells to sample.
    seed: `int`
        Seed for random number generator.

    Returns:
    --------
    src_adata: `~anndata.AnnData`
        Source annotated data matrix with subsetted cells.
    tgt_adata: `~anndata.AnnData`
        Target annotated data matrix with subsetted cells.
    src_dataset: `~datasets.DatasetDict`
        Source dataset with subsetted cells.
    tgt_dataset: `~datasets.DatasetDict`
        Target dataset with subsetted cells.
    """
    np.random.seed(seed)
    if num_cells != 0:
        # choose from newly enumerated index for obs
        indices_range = range(src_adata.shape[0])
        sample_idx = np.random.choice(indices_range, num_cells, replace=False)
        src_adata = src_adata[sample_idx, :]
        tgt_adata = tgt_adata[sample_idx, :]
        src_dataset = src_dataset.select(sample_idx)
        tgt_dataset = tgt_dataset.select(sample_idx)
    return src_adata, tgt_adata, src_dataset, tgt_dataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tokenid_mapping(
    adata: ad.AnnData,
    token_id_path: str,
    exclude_non_GF_genes: bool = False,
):
    with open(token_id_path, 'rb') as f:
        token_id_dict = pickle.load(f)
    adata.var['token_id'] = adata.var_names.map(token_id_dict)
    adata.var['token_id'] = adata.var['token_id'].astype('Int64')
    if exclude_non_GF_genes:
        adata_subset = adata[:, adata.var['token_id'].notna()].copy()
        print(f'Number of genes dropped: {adata.n_vars - adata_subset.n_vars}')
    else:
        adata_subset = adata.copy()
    # keep special tokens and do not assign row_id
    pattern = re.compile(r'<[a-zA-Z]+>')
    special_tokens = [s for s in token_id_dict.keys() if pattern.match(s)]
    special_tokens_dict = {
        key: value for key, value in token_id_dict.items() if key in special_tokens
    }
    # create row_id for special tokens and exclude special tokens from row_id
    row_id = np.arange(adata_subset.n_vars + len(special_tokens))
    map_special_tokens = dict(
        zip(special_tokens_dict.values(), special_tokens_dict.values())
    )
    row_id = row_id[~np.isin(row_id, list(special_tokens_dict.values()))]
    adata_subset.var['row_id'] = row_id
    # create dictionary to map token_id to row_id
    token_id_to_row_id_dict = dict(
        zip(
            adata_subset.var['token_id'].values,
            adata_subset.var['row_id'].values,
        )
    )
    # add token_id_row_id_dict for special tokens
    token_id_to_row_id_dict.update(map_special_tokens)
    # create dictionary to map row_id to gene_name
    row_id_to_gene_name = dict(
        zip(adata_subset.var['row_id'], adata_subset.var['gene_name'])
    )
    special_tokens_dict = {value: key for key, value in special_tokens_dict.items()}
    row_id_to_gene_name.update(special_tokens_dict)
    return (adata_subset, token_id_to_row_id_dict, row_id_to_gene_name)


# use dictionary to map token_id to input_ids
def map_input_ids_to_row_id(
    dataset: DatasetDict,
    token_id_to_row_id_dict: Dict,
    ignore_tokens: Optional[List] = None,
):
    if ignore_tokens is None:
        ignore_tokens = []
    dataset['input_ids'] = [
        token_id_to_row_id_dict.get(item, item)
        for item in dataset['input_ids']
        if item not in ignore_tokens
    ]
    return dataset


def subset_adata(adata, cell_pairings):
    adata_ = adata.copy()
    # check if obs index is not range index
    if adata_.obs.index.dtype != 'int64':
        adata_.obs = adata_.obs.reset_index()
    df = pd.DataFrame(
        adata_.X.toarray(), index=adata_.obs.index, columns=adata_.var.index
    )
    # use row index instead of index
    df.reset_index(drop=True, inplace=True)
    subset_df = df.loc[cell_pairings]
    adata_obs_subsetted = adata_.obs.loc[cell_pairings]
    obs = adata_obs_subsetted
    var = adata_.var.loc[df.columns]
    adata_subsetted = ad.AnnData(
        X=subset_df.values,
        obs=obs,
        var=var,
    )
    adata_subsetted.obs_names.name = None
    adata_subsetted.X = csr_matrix(adata_subsetted.X)
    return adata_subsetted


# Code adapte from lucidrains/muse-maskgit-pytorch
# https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/muse_maskgit_pytorch.py#L26 # noqa

# generation helper functions


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(min, max)


def noise_schedule(
    ratio,
    method,
    exponent=2.0,
    total_tokens=None,
):
    '''
    Noise schedule from Google MaskGIT paper
    URL: https://github.com/google-research/maskgit/blob/1db23594e1bd328ee78eadcd148a19281cd0f5b8/maskgit/libml/mask_schedule.py#L21 # noqa
    Last accessed: 2024-03-23
    '''
    if method == 'uniform':
        mask_ratio = 1.0 - ratio
    elif 'pow' in method:
        mask_ratio = 1.0 - ratio**exponent
    elif method == 'cosine':
        mask_ratio = torch.cos(ratio * math.pi * 0.5)
    elif method == 'log':
        mask_ratio = -torch.log2(ratio) / torch.log2(total_tokens)
    elif method == 'exp':
        mask_ratio = 1 - torch.exp2(-torch.log2(total_tokens) * (1 - ratio))
    # Clamps mask into [epsilon, 1)
    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.0)
    return mask_ratio


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


# sampling helper
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def mean_nonpadding_embs(
    embs: torch.Tensor,
    input_ids: torch.Tensor,
    mapping_dict: dict,
    condition_dict: dict,
    dim: int = 1,
    perturbation_tokens: torch.Tensor | None = None,
    perturbation_mode: Literal['mask', 'pad', 'delete', 'overexpress'] | None = None,
):
    '''
    Compute the mean of the non-padding embeddings.
    Modified from Geneformer:
    https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py # noqa
    Accessed: 2024-05-14
    '''
    # create a mask to exclude special and perturbation tokens
    special_token_names = ['<cls>', '<mask>', '<pad>', '<eos>']
    special_tokens = [mapping_dict[token] for token in special_token_names]
    if condition_dict is not None:
        cond_tokens = []
        for condition in condition_dict:
            cond_tokens.extend(list(condition_dict[condition].values()))
        special_tokens.extend(cond_tokens)
    special_tokens = torch.tensor(special_tokens, device=embs.device)
    if perturbation_mode is not None:
        if perturbation_mode in ['delete', 'overexpress']:
            # do not mask tokens as they are deleted
            perturbation_tokens = None

    if perturbation_tokens is not None:
        tokens_to_exclude = torch.cat([special_tokens, perturbation_tokens])
    else:
        tokens_to_exclude = special_tokens

    pad_mask = torch.isin(input_ids, tokens_to_exclude)
    if (perturbation_mode is not None) and (perturbation_mode == 'overexpress'):
        # mask overexpressed tokens at first position
        pad_mask[:, 0] = True
    # our mask is the opposite of BERT mask
    pad_mask = ~pad_mask
    # create a tensor of original lengths
    original_lens = pad_mask.sum(dim=1)

    # create CLS token mask
    if embs.dim() == 3:
        # fill the masked positions in embs with zeros
        masked_embs = embs.masked_fill(~pad_mask.unsqueeze(2), 0.0)
        # compute the mean across the non-padding dimensions
        mean_embs = masked_embs.sum(dim) / original_lens.view(-1, 1).float()

    elif embs.dim() == 2:
        masked_embs = embs.masked_fill(~pad_mask, 0.0)
        mean_embs = masked_embs.sum(dim) / original_lens.float()
    return mean_embs


def generate_pad(input_ids):
    '''
    Description:
    ------------
    Generate padding mask for target tensor.
    For tgt tensor, pad token is 0 and non-pad token is 1.
    Convert tgt tensor to boolean tensor,
    where pad token is True and non-pad token is False.
    Can also be applied to generate source padding mask.
    '''
    pad = input_ids == 0
    return pad


def pairing_src_to_tgt_cells(
    adata_subset: sc.AnnData,
    pairing_mode: Literal['random', 'stratified'],
    pairing_obs: str,
    seed_no: int = 42,
    mapping_df=Optional[pd.DataFrame],
):
    """
    Description:
    ------------
    This function pairs resting cells to activated cells based on time point.

    Parameters:
    -----------
    adata_subset: `~anndata.AnnData`
        Annotated data matrix subsetted to include only DEGs.
    pairing_mode: `str`
        Mode to pair cells. Choose between 'random' and 'stratified'.
    seed: `int`
        Seed for random number generator.

    Returns:
    --------
    cell_pairings: `dict`
        Dictionary containing pairing indices of resting and activated cells.
    """
    np.random.seed(seed_no)
    # replace index by row number
    adata_subset_ = adata_subset.copy()
    adata_subset_.obs = adata_subset_.obs.reset_index()
    adata_dict = {}
    # initiate dict to store cell pairing
    cell_pairings: Dict[str, List[str]] = {}
    max_rows = 0
    max_reference_time = None
    for category in adata_subset_.obs[pairing_obs].unique():
        adata_dict[category] = adata_subset_.obs.loc[
            adata_subset_.obs[pairing_obs] == category, :
        ]
        cell_pairings[category] = []
        # Check if this category has more rows than the current maximum
        if len(adata_dict[category]) > max_rows:
            max_rows = len(adata_dict[category])
            max_reference_time = category
    if pairing_mode == 'stratified':
        # drop Donor if they do not have Cell_type, Donor in all the Time_points
        adata_grouped = adata_subset_.obs[
            adata_subset_.obs.groupby(['Donor', 'Cell_type'])[pairing_obs].transform(
                'nunique'
            )
            == 4
        ]
        dropped_donors = (
            adata_subset.obs['Donor'].nunique() - adata_grouped['Donor'].nunique()
        )
        print(f'dropped {dropped_donors} donors')
        resting_cells = adata_grouped.loc[adata_grouped[pairing_obs] == '0h', :]
        grouped = adata_grouped.groupby(['Donor', 'Cell_type'])
        for idx, resting in tqdm.tqdm(
            resting_cells.iterrows(), total=resting_cells.shape[0]
        ):
            # get the indices of the other time points for the same cell type and donor
            group = grouped.get_group((resting['Donor'], resting['Cell_type']))
            indices_16h = group[group[pairing_obs] == '16h'].index
            indices_40h = group[group[pairing_obs] == '40h'].index
            indices_5d = group[group[pairing_obs] == '5d'].index
            cell_pairings['0h'].append(idx)
            cell_pairings['16h'].append(np.random.choice(indices_16h))
            cell_pairings['40h'].append(np.random.choice(indices_40h))
            cell_pairings['5d'].append(np.random.choice(indices_5d))
    if pairing_mode == 'mapping':
        for condition in mapping_df[max_reference_time].unique():
            mapping_df_ = mapping_df[mapping_df[max_reference_time] == condition]
            adata_ = adata_dict[max_reference_time]
            cell_to_pair = adata_['celltype_v2'][
                adata_['celltype_v2'].isin(mapping_df_[max_reference_time])
            ].index
            cell_pairings[max_reference_time].extend(cell_to_pair)
            n_cells_to_pair = len(cell_to_pair)

            for stage, adata_ in adata_dict.items():
                if stage != max_reference_time:
                    cell_to_pair = adata_['celltype_v2'][
                        adata_['celltype_v2'].isin(mapping_df_[stage])
                    ].index
                    # only sample with replacement if needed
                    if n_cells_to_pair > cell_to_pair.shape[0]:
                        sample_with_replacement = True
                    else:
                        sample_with_replacement = False
                    cell_pairings[stage].extend(
                        np.random.choice(
                            cell_to_pair,
                            n_cells_to_pair,
                            replace=sample_with_replacement,
                        )
                    )
                else:
                    continue
    elif pairing_mode == 'random':
        if max_reference_time is not None:
            # randomly sample from each time point
            ref_adata = adata_dict[max_reference_time]
            cell_pairings[max_reference_time] = ref_adata.index.tolist()
            # remove reference time from dictionary
            del adata_dict[max_reference_time]
            for rest_time, adata_ in adata_dict.items():
                cell_pairings[rest_time] = np.random.choice(
                    adata_.index, len(ref_adata), replace=True
                ).tolist()
    else:
        raise ValueError('pairing_mode must be either random or stratified')
    return cell_pairings


def label_encoder(adata, encoder, condition_key=None) -> np.ndarray:
    """
    Description:
    ------------
    Encode labels of Annotated `adata` matrix.

    Parameters:
    ----------
    adata: : `~anndata.AnnData`
         Annotated data matrix.
    encoder: Dict
         dictionary of encoded labels.
    condition_key: String
         column name of conditions in `adata.obs` data frame.

    Returns:
    -------
    labels: `~numpy.ndarray`
         Array of encoded labels
    label_encoder: Dict
         dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(encoder.keys())):
        missing_labels = set(unique_conditions).difference(set(encoder.keys()))
        print(
            f'Warning: Labels in adata.obs[{condition_key}]'
            'is not a subset of label-encoder!'
        )
        print(f'The missing labels are: {missing_labels}')
        print('Therefore integer value of those labels is set to -1')
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    labels = [int(x) for x in labels]
    return labels


def randomised_split(adata: ad.AnnData, train_prop: float, test_prop: float, seed: int):
    n_cells = adata.shape[0]
    indices = np.arange(n_cells)
    # define train, val and test size
    train_size = np.round(train_prop * n_cells).astype(int)
    test_size = np.round(test_prop * n_cells).astype(int)
    # val_size = adata.shape - train_size - test_size
    # generator = torch.Generator().manual_seed(seed)
    # train, val, test = random_split(
    #     dataset, [train_size, val_size, test_size], generator=generator
    # )
    train_indices = np.random.choice(indices, train_size, replace=False)

    indices_ = np.setdiff1d(indices, train_indices)

    test_indices = np.random.choice(indices_, test_size, replace=False)
    indices_ = np.setdiff1d(indices_, test_indices)
    val_indices = indices_

    return train_indices, val_indices, test_indices


def stratified_split(
    tgt_adata: ad.AnnData,
    train_prop: float,
    test_prop: float,
    groups: List[str],
    seed: int = 42,
):
    """
    Description:
    ------------
    Stratified split of dataset based on cell type.
    """
    np.random.seed(seed)
    # define train, val and test size based on unique groups
    # extract unique groups and counts
    # groups =
    groups_df = tgt_adata.obs[groups].copy()
    if len(groups) > 1:
        groups_df.loc[:, 'stratified'] = groups_df.loc[:, groups].apply(
            lambda x: '_'.join(x), axis=1
        )
    else:
        groups_df.loc[:, 'stratified'] = groups_df.loc[:, groups]
    groups_df.reset_index(drop=True, inplace=True)
    unique_groups = groups_df['stratified'].unique()
    group_indices = [np.where(groups_df['stratified'] == i)[0] for i in unique_groups]
    train_indices, test_indices, val_indices = [], [], []

    for indices in group_indices:
        assert (
            len(np.unique(groups_df.iloc[indices].stratified)) == 1
        ), 'groups are not stratified'
        # split indices into train, val and test set
        np.random.shuffle(indices)
        train_size = np.round(train_prop * len(indices)).astype(int)
        test_size = np.round(test_prop * len(indices)).astype(int)
        # val_size = len(indices) - train_size - test_size
        train_indices.extend(indices[:train_size])
        test_indices.extend(indices[train_size : train_size + test_size])
        val_indices.extend(indices[train_size + test_size :])
    return train_indices, val_indices, test_indices


def unseen_donor_split(
    adata: ad.AnnData,
    train_prop: float,
    test_prop: float,
):
    # define groups for stratified split by Time_point and Cell_type
    groups = adata.obs[['Donor']]
    # define train, val and test size based on unique donors
    train_size = np.round(train_prop * len(groups['Donor'].unique())).astype(int)
    test_size = np.round(test_prop * len(groups['Donor'].unique())).astype(int)
    val_size = len(groups['Donor'].unique()) - train_size - test_size
    # sample from groups based on unique donors using numpy random choice
    test_donors = np.random.choice(
        groups['Donor'].unique(), size=test_size, replace=False
    )
    # exclude test donors from train and val set
    train_val_donors = np.setdiff1d(groups['Donor'].unique(), test_donors)
    # sample from remaining donors based on unique donors using numpy random choice
    val_donors = np.random.choice(train_val_donors, size=val_size, replace=False)
    # use remaining donors as train set
    train_donors = np.setdiff1d(train_val_donors, val_donors)
    # split dataset to create dataset subset not tuple
    # get indices of train, val and test set
    train = Subset(adata, np.where(groups['Donor'].isin(train_donors))[0])
    val = Subset(adata, np.where(groups['Donor'].isin(val_donors))[0])
    test = Subset(adata, np.where(groups['Donor'].isin(test_donors))[0])

    return train, val, test


def gen_attention_mask(self, length, max_len=1000):
    attention_mask = [
        [1] * original_len + [0] * (max_len - original_len)
        if original_len <= max_len
        else [1] * max_len
        for original_len in length
    ]

    return torch.tensor(attention_mask)


# inherit EmbExtractor from Geneformer to avoid sorting of embs
class non_sorted_EmbExtractor(EmbExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_embs(
        self,
        model_directory,
        input_data_file,
        output_directory,
        output_prefix,
        output_torch_embs=False,
        cell_state=None,
    ):
        filtered_input_data = pu.load_and_filter(
            self.filter_data, self.nproc, input_data_file
        )
        if cell_state is not None:
            filtered_input_data = pu.filter_by_dict(
                filtered_input_data, cell_state, self.nproc
            )
        model = pu.load_model(self.model_type, self.num_classes, model_directory)
        layer_to_quant = pu.quant_layers(model) + self.emb_layer
        embs = get_embs(
            model,
            filtered_input_data,  # Remove downsampling code
            self.emb_mode,
            layer_to_quant,
            self.pad_token_id,
            self.forward_batch_size,
            self.summary_stat,
        )

        if self.emb_mode == 'cell':
            if self.summary_stat is None:
                embs_df = label_cell_embs(embs, filtered_input_data, self.emb_label)
            elif self.summary_stat is not None:
                embs_df = pd.DataFrame(embs.cpu().numpy()).T
        elif self.emb_mode == 'gene':
            if self.summary_stat is None:
                embs_df = self.label_gene_embs(
                    embs, filtered_input_data, self.token_gene_dict
                )
            elif self.summary_stat is not None:
                embs_df = pd.DataFrame(embs).T
                embs_df.index = [self.token_gene_dict[token] for token in embs_df.index]

        # save embeddings to output_path
        if cell_state is None:
            output_path = (Path(output_directory) / output_prefix).with_suffix('.csv')
            embs_df.to_csv(output_path)

        if self.exact_summary_stat == 'exact_mean':
            embs = embs.mean(dim=0)
            embs_df = pd.DataFrame(
                embs_df[0:255].mean(axis='rows'), columns=[self.exact_summary_stat]
            ).T
        elif self.exact_summary_stat == 'exact_median':
            embs = torch.median(embs, dim=0)[0]
            embs_df = pd.DataFrame(
                embs_df[0:255].median(axis='rows'), columns=[self.exact_summary_stat]
            ).T
        if cell_state is not None:
            return embs
        else:
            if output_torch_embs:
                return embs_df, embs
            else:
                return embs_df


# class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#             self,
#             optimizer: Optimizer,
#             warmup_steps: int,
#             initial_lr: float,
#             end_lr: float,
#             last_epoch: int = -1
#             ):
#         self.warmup_steps = warmup_steps
#         self.initial_lr = initial_lr
#         self.end_lr = end_lr
#         super(WarmupScheduler, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         current_step = self.last_epoch + 1

#         if current_step < self.warmup_steps:
#             # Linear warmup phase: increase from initial_lr to end_lr
#             warmup_lr = [
#                 self.initial_lr +
#                 (self.end_lr - self.initial_lr) * (current_step / self.warmup_steps)
#                 for _ in self.base_lrs
#             ]
#             return warmup_lr
#         else:
#             # After warmup, maintain the end_lr (constant LR)
#             return [self.end_lr for _ in self.base_lrs]
