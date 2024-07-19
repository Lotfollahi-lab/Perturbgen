import argparse
import math
import os
import pickle
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
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
from torch.utils.data import Subset, WeightedRandomSampler
from torchmetrics import PearsonCorrCoef


def read_dataset_files(directory: str, file_type: str):
    '''
    Description:
    ------------
    Read dataset files from a directory and
    return a dictionary of datasets amd max input_id.



    '''
    dataset_dict = {}
    for filename in os.listdir(directory):
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
            elif file_type == 'pkl':
                print(filename)
                with open(filename_, 'rb') as f:
                    dataset_dict[f'tgt_{file_type}_t{filename[0]}'] = pickle.load(f)
            else:
                raise ValueError(f'{file_type} must be either dataset or h5ad')
    return dataset_dict


def mean_nonpadding_embs(embs, pad, dim=1):
    '''
    Compute the mean of the non-padding embeddings.
    Modified from Geneformer:
    https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py # noqa
    Accessed: 2024-05-14
    '''
    # mask should be opposite of pad
    pad[:, 0] = True
    # our mask is the opposite of BERT mask
    pad_mask = ~pad
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


def compute_cos_similarity(
    outputs: dict,
    time_step: Optional[int] = None,
    all_time_steps: Optional[List[int]] = None,
):
    """
    Description:
    ------------
    This function computes cosine similarity between cls and gene embeddings.

    Parameters:
    -----------
    outputs: `dict`
        Dictionary containing outputs from the model.
    time_step: `Optional[int]`
        Time step to compute cosine similarity.
    all_time_steps: `Optional[List[int]`
        List of all time steps.

    Returns:
    --------
    cos_similarity: `torch.tensor`
        Tensor of cosine similarity between cls and gene embeddings.
    cls_embeddings: `torch.tensor`
    gene_embeddings: `torch.tensor`
    """
    if time_step is not None:
        # check if cls position is in outputs
        assert 'cls_positions' in outputs.keys(), 'cls position not in outputs'
        assert 'dec_embedding' in outputs.keys(), 'dec_embedding not in outputs'
        # get cls position and dec_embedding (index = time_step-1)
        cls_position = outputs['cls_positions'][time_step - 1]
        cls_embeddings = outputs['dec_embedding'][:, cls_position, :]
        # exclude cls token from gene embeddings
        if (time_step is not None) and (all_time_steps is not None):
            if time_step == max(all_time_steps):
                gene_embeddings = outputs['dec_embedding'][:, (cls_position + 1) :, :]
            else:
                gene_embeddings = outputs['dec_embedding'][
                    :, (cls_position + 1) : outputs['cls_positions'][time_step], :
                ]
        cos_similarity = []
        for i in range(gene_embeddings.shape[0]):
            # gene level cosine similarity
            cos_similarity_ = cosine_similarity(
                cls_embeddings[i],
                gene_embeddings[i, :, :],
                dim=1,
            )
            cos_similarity.append(cos_similarity_)
        cos_similarity = torch.stack(cos_similarity)
    else:
        cls_embeddings = outputs['mean_embedding']
        gene_embeddings = outputs['dec_embedding'][:, 1:, :]
        cos_similarity = []
        for i in range(gene_embeddings.shape[0]):
            # gene level cosine similarity
            cos_similarity_ = cosine_similarity(
                cls_embeddings[i],
                gene_embeddings[i, :, :],
                dim=1,
            )
            cos_similarity.append(cos_similarity_)
        cos_similarity = torch.stack(cos_similarity)

    return cos_similarity, cls_embeddings, gene_embeddings


def return_cos_similarity(
    marker_genes: List[str],
    cos_similarity: torch.tensor,
    gene_embeddings: torch.tensor,
    mapping_dict: Dict,
    token_ids: torch.tensor,
) -> torch.tensor:
    """
    Description:
    ------------
    This function returns cosine similarity for marker genes.

    Parameters:
    -----------
    marker_genes: `List[str]`
        List of marker genes.
    cos_similarity: `torch.tensor`
        Tensor of cosine similarity between cls and gene embeddings.
    gene_embeddings: `torch.tensor`
        Tensor of gene embeddings.
    mapping_dict: `Dict`
        Dictionary mapping gene names to token ids.

    Returns:
    --------
    cos_similarity_res: `torch.tensor`
        Tensor of cosine similarity for marker genes.
    marker_genes_dict: `Dict`
    """
    # filter for marker genes and swap key value
    marker_genes_ids = {v: k for k, v in mapping_dict.items() if v in marker_genes}
    cos_similarity_res = torch.zeros(
        cos_similarity.shape[0],
        len(marker_genes_ids.keys()),
        device=gene_embeddings.device,
    )
    marker_genes_dict = {}
    for i, gene in enumerate(marker_genes_ids.keys()):
        # extract cosine similarity for marker genes
        # ---------------------
        cond_embs_to_fill = (token_ids == marker_genes_ids[gene]).sum(1) > 0
        cond_select_markers = torch.where(token_ids == marker_genes_ids[gene])
        cos_similarity_res[cond_embs_to_fill, i] = cos_similarity[
            cond_select_markers[0], cond_select_markers[1]
        ]
        marker_genes_dict[gene] = i
    return cos_similarity_res, marker_genes_dict


def return_gene_embeddings(
    marker_genes: List[str],
    gene_embeddings: torch.tensor,
    mapping_dict: Dict,
    token_ids: torch.tensor,
) -> torch.tensor:
    """
    Description:
    ------------
    This function returns gene embeddings from a list of marker genes.

    Parameters:
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
    # filter for marker genes and swap key value
    marker_genes_ids = {v: k for k, v in mapping_dict.items() if v in marker_genes}
    gene_embeddings_res = torch.zeros(
        gene_embeddings.shape[0],
        len(marker_genes_ids.keys()),
        gene_embeddings.shape[2],
        device=gene_embeddings.device,
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


def modify_ckpt_state_dict(
    checkpoint: dict,
    replace_str: str,
):
    """
    Description:
    ------------
    This function modifies the state_dict of a checkpoint
    by removing the replace_str from the keys.

    Parameters:
    -----------
    checkpoint: `dict`
        Checkpoint dictionary loaded using torch.load.
    replace_str: `str`
        String to replace in the keys.

    Returns:
    --------
    new_state_dict: `dict`
        Modified state_dict.
    """
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
    pearson = PearsonCorrCoef(num_outputs=num_outputs).to('cuda')
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


def noise_schedule(
    ratio: float, total_tokens: int, method: str, exponent: float = 2.0
) -> torch.Tensor:
    '''
    Description:
    ------------
    Noise schedule from Google MaskGIT paper
    URL: https://github.com/google-research/maskgit/blob/1db23594e1bd328ee78eadcd148a19281cd0f5b8/maskgit/libml/mask_schedule.py#L21 # noqa
    Last accessed: 2024-03-23

    Parameters:
    -----------
    ratio: `float`
        Ratio of tokens to mask.
    total_tokens: `int`
        Total number of tokens.
    method: `str`
        Method to compute mask ratio.
        Options: 'uniform', 'pow', 'cosine', 'log', 'exp'.
    exponent: `float`
        Exponent for 'pow' method.
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


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(min, max)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


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


def generate_pad(tgt):
    '''
    Description:
    ------------
    Generate padding mask for target tensor.
    For tgt tensor, pad token is 0 and non-pad token is 1.
    Convert tgt tensor to boolean tensor,
    where pad token is True and non-pad token is False.
    Can also be applied to generate source padding mask.
    '''
    tgt_pad = tgt == 0
    return tgt_pad


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
    # print number of genes dropped
    print(f'Number of genes dropped: {adata.n_vars - adata_subset.n_vars}')
    adata_subset.var['row_id'] = np.arange(adata_subset.n_vars) + 1
    # create dictionary to map token_id to row_id
    token_id_to_row_id_dict = dict(
        zip(
            adata_subset.var['token_id'].values,
            adata_subset.var['row_id'].values,
        )
    )
    token_id_to_row_id_dict[0] = 0
    # create dictionary to map row_id to gene_name
    row_id_to_gene_name = dict(
        zip(adata_subset.var['row_id'], adata_subset.var['gene_name'])
    )
    return (adata_subset, token_id_to_row_id_dict, row_id_to_gene_name)


# use dictionary to map token_id to input_ids
def map_input_ids_to_row_id(dataset, token_id_to_row_id_dict):
    dataset['input_ids'] = [
        token_id_to_row_id_dict.get(item, item) for item in dataset['input_ids']
    ]
    return dataset


def subset_adata(adata, cell_pairings):
    adata_ = adata.copy()
    # check if obs index is not range index
    if adata_.obs.index.dtype != 'int64':
        adata_.obs = adata_.obs.reset_index()
    df = pd.DataFrame(adata_.X.A, index=adata_.obs.index, columns=adata_.var.index)
    # use row index instead of index
    df.reset_index(drop=True, inplace=True)
    subset_df = df.loc[cell_pairings]
    adata_obs_subsetted = adata_.obs.loc[cell_pairings]
    obs = adata_obs_subsetted
    var = adata_.var.loc[df.columns]
    adata_subsetted = ad.AnnData(X=subset_df.values, obs=obs, var=var)
    adata_subsetted.obs_names.name = None
    adata_subsetted.X = csr_matrix(adata_subsetted.X)
    return adata_subsetted


def pairing_src_to_tgt_cells(
    adata_subset: ad.AnnData,
    pairing_mode: str,
    pairing_obs: str,
    dataset_type: str,
    seed_no: int = 42,
    src_condition: Optional[List[str]] = None,
    tgt_conditions: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Description:
    ------------
    This function pairs control cells to perturbed cells
    can be either single or multiple discrete timepoint.

    Parameters:
    -----------
    adata_subset: `~anndata.AnnData`
        Annotated data matrix subsetted to include only DEGs.
    pairing_mode: `str`
        Mode to pair cells. Choose between 'random' and 'stratified'.
    dataset_type: `str`
        Type of dataset. Choose between 'conditional' and 'temporal'.
    pairing_obs: `str`
        obs column name in anndate which is used as pairing condition
    seed: `int`
        Seed for random number generator.
    src_condition: `Optional[List[str]]`
        Source/control condition for cell pairing.
        Select a category in pairing_obs column.
    tgt_conditions: `Optional[List[str]]`
        Target/perturbed condition for cell pairing.
        Select categories in pairing_obs column.


    Returns:
    --------
    cell_pairings: `Dict[str, List[str]]`
        Dictionary containing pairing indices of control and perturbed cells.
    """
    np.random.seed(seed_no)
    # initiate dict to store condition specific adata
    adata_dict = {}
    # initiate dict to store cell pairing
    cell_pairings: Dict[Any, Any] = {}
    if pairing_mode == 'stratified':
        # replace index by row number
        adata_subset_ = adata_subset.copy()
        adata_subset_.obs = adata_subset_.obs.reset_index()
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

    elif pairing_mode == 'random':
        max_reference_cond = ''
        if src_condition is not None:
            for condition in src_condition:
                adata_dict[condition] = adata_subset.obs.loc[
                    adata_subset.obs[pairing_obs] == condition, :
                ]
                adata_dict[condition] = adata_dict[condition].reset_index()
        if tgt_conditions is not None:
            for condition in tgt_conditions:
                adata_dict[condition] = adata_subset.obs.loc[
                    adata_subset.obs[pairing_obs] == condition, :
                ]
                adata_dict[condition] = adata_dict[condition].reset_index()
                if (
                    (dataset_type == 'conditional')
                    and (tgt_conditions is not None)
                    and (src_condition is not None)
                ):
                    if condition in tgt_conditions:
                        tgt_indices = adata_dict[condition].index.tolist()
                        tmp_cell_no = len(tgt_indices)
                        mapping_name = f'{src_condition[0]}_{condition}'
                        src_indices = np.random.choice(
                            adata_dict[src_condition[0]].index,
                            tmp_cell_no,
                            replace=True,
                        ).tolist()
                        cell_pairings[mapping_name] = dict(
                            zip(src_indices, tgt_indices)
                        )
                    elif condition in src_condition:
                        pass
                    else:
                        raise ValueError(f'{condition} not in {pairing_obs}')
                # Check if this adata_tmp has more rows than the current maximum
                elif dataset_type == 'temporal':
                    tgt_cell_no = len(adata_dict[condition])
                    max_reference_cond = condition

        if dataset_type == 'temporal':
            # randomly sample from each condition
            ref_adata = adata_dict[max_reference_cond]
            cell_pairings[max_reference_cond] = ref_adata.index.tolist()
            for rest_time, adata_ in adata_dict.items():
                if rest_time != max_reference_cond:
                    cell_pairings[rest_time] = np.random.choice(
                        adata_.index, tgt_cell_no, replace=True
                    ).tolist()
            # remove reference time from dictionary
            del adata_dict[max_reference_cond]
            for rest_time, adata_ in adata_dict.items():
                cell_pairings[rest_time] = np.random.choice(
                    adata_.index, tgt_cell_no, replace=True
                ).tolist()
        elif dataset_type == 'conditional':
            pass
        else:
            raise ValueError('dataset_type must be either conditional or temporal')
    else:
        raise ValueError('pairing_mode must be either random or stratified')
    return cell_pairings


def weighted_sampler(dataset_list):
    '''
    Description:
    ------------
    Create weighted sampler without replacement for dataloader based on dataset lengths.
    Samples are weighted based on the inverse of dataset lengths to balance the dataset.

    Parameters:
    -----------
    dataset_list: `List[datasets.Dataset]`
        List of datasets.
    '''
    lengths = [len(dataset) for dataset in dataset_list]
    weights = [1.0 / length for length in lengths for _ in range(length)]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=False
    )

    return sampler


def filter_dataset(
    dataset: DatasetDict, condition_key: str, condition_values: List[str]
):
    '''
    Description:
    ------------
    Filter Huggingface dataset based on condition values.

    Parameters:
    -----------
    dataset: `datasets.DatasetDict`
        Huggingface dataset.
    condition_key: `str`
        Column name of conditions in dataset.
    condition_values: `List[str]`
        Condition values to filter dataset which are present in condition_key.
    '''
    return dataset.filter(lambda x: x[condition_key] in condition_values)


def label_encoder(adata, encoder, condition_key=None):
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
    np.random.seed(seed)
    n_cells = adata.shape[0]
    indices = np.arange(n_cells)
    # define train, val and test size
    train_size = np.round(train_prop * n_cells).astype(int)
    test_size = np.round(test_prop * n_cells).astype(int)
    train_indices = np.random.choice(indices, train_size, replace=False)
    indices_ = np.setdiff1d(indices, train_indices)
    test_indices = np.random.choice(indices_, test_size, replace=False)
    indices_ = np.setdiff1d(indices_, test_indices)
    val_indices = indices_
    return train_indices, val_indices, test_indices


def randomised_mapping_dir_split(
    mapping_dir: Dict[int, int],
    train_prop: float,
    test_prop: float,
    seed: int,
) -> tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    np.random.seed(seed)
    n_cells = len(mapping_dir)
    # define train, val and test size
    train_size = int(train_prop * n_cells)
    test_size = int(test_prop * n_cells)
    # check uniqueness of key and values to create train-test split datasets
    if len(set(mapping_dir.keys())) == len(mapping_dir):
        indices = list(mapping_dir.keys())
        index_type = 'keys'
    elif len(set(mapping_dir.values())) == len(mapping_dir):
        indices = list(mapping_dir.values())
        index_type = 'values'
    else:
        raise ValueError('Dictionary keys and values are not unique')

    train_indices = np.random.choice(indices, train_size, replace=False)
    indices_ = np.setdiff1d(indices, train_indices)
    test_indices = np.random.choice(indices_, test_size, replace=False)
    indices_ = np.setdiff1d(indices_, test_indices)
    val_indices = indices_

    if index_type == 'keys':
        train_dict = {k: v for k, v in mapping_dir.items() if k in train_indices}
        val_dict = {k: v for k, v in mapping_dir.items() if k in val_indices}
        test_dict = {k: v for k, v in mapping_dir.items() if k in test_indices}
    elif index_type == 'values':
        train_dict = {k: v for k, v in mapping_dir.items() if v in train_indices}
        val_dict = {k: v for k, v in mapping_dir.items() if v in val_indices}
        test_dict = {k: v for k, v in mapping_dir.items() if v in test_indices}
    return train_dict, val_dict, test_dict


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
