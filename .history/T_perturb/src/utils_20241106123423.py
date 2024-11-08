import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List

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
from torch.utils.data import Subset
from torchmetrics import PearsonCorrCoef
from torch.optim import Optimizer

def read_dataset_files(directory, file_type):
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


def compute_cos_similarity(
    outputs: dict,
    time_step: int,
    all_time_steps: list[int],
):
    """
    Description:
    ------------
    This function computes cosine similarity between cls and gene embeddings.
    Parameters:
    -----------
    outputs: `dict`
        Dictionary containing outputs from the model.
    time_step: `int`
        Time step to compute cosine similarity.
    all_time_steps: `list[int]`
        List of all time steps.
    Returns:
    --------
    cos_similarity: `torch.tensor`
        Tensor of cosine similarity between cls and gene embeddings.
    cls_embeddings: `torch.tensor`
    gene_embeddings: `torch.tensor`
    """
    # get cls position and dec_embedding (index = time_step-1)
    dec_embedding = outputs['dec_embedding'][time_step]
    cls_embeddings = dec_embedding[:, 0, :]
    # exclude cls token from gene embeddings
    gene_embeddings = dec_embedding[:, 1:, :]
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


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(min, max)


def mean_nonpadding_embs(embs, pad, dim=1):
    '''
    Compute the mean of the non-padding embeddings.
    Modified from Geneformer:
    https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py # noqa
    Accessed: 2024-05-14
    '''
    pad_mask = pad.clone()
    # mask should be opposite of pad
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


def pairing_src_to_tgt_cells(
    adata_subset: sc.AnnData,
    pairing_mode: str,
    pairing_obs: str,
    seed_no: int = 42,
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
    for adata_tmp in adata_subset_.obs[pairing_obs].unique():
        adata_dict[adata_tmp] = adata_subset_.obs.loc[
            adata_subset_.obs[pairing_obs] == adata_tmp, :
        ]
        cell_pairings[adata_tmp] = []
        # Check if this adata_tmp has more rows than the current maximum
        if len(adata_dict[adata_tmp]) > max_rows:
            max_rows = len(adata_dict[adata_tmp])
            max_reference_time = adata_tmp
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
    n_cells = adata.shape[0]
    indices = np.arange(n_cells)
    print(len(indices))

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


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, initial_lr: float, end_lr: float, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1

        if current_step < self.warmup_steps:
            # Linear warmup phase: increase from initial_lr to end_lr
            warmup_lr = [
                self.initial_lr + (self.end_lr - self.initial_lr) * (current_step / self.warmup_steps) 
                for _ in self.base_lrs
            ]
            return warmup_lr
        else:
            # After warmup, maintain the end_lr (constant LR)
            return [self.end_lr for _ in self.base_lrs]