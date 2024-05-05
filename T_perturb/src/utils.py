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
from torch.utils.data import Subset


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


def tokenid_mapping(adata: ad.AnnData, token_id_path: str):
    with open(token_id_path, 'rb') as f:
        token_id_dict = pickle.load(f)
    adata.var['token_id'] = adata.var_names.map(token_id_dict)
    adata.var['token_id'] = adata.var['token_id'].astype('Int64')
    adata_subset = adata[:, adata.var['token_id'].notna()].copy()
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


def pairing_resting_to_activated_cells(
    adata_subset: sc.AnnData,
    pairing_mode: str,
    seed_no: int = 42,
    reference_time: str = '0h',
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
    cell_pairings: Dict[str, List[int]] = {}
    for adata_tmp in adata_subset_.obs['Time_point'].unique():
        adata_dict[adata_tmp] = adata_subset_.obs.loc[
            adata_subset_.obs['Time_point'] == adata_tmp, :
        ]
        cell_pairings[adata_tmp] = []
    # initiate dictionary to store cell pairings

    if pairing_mode == 'stratified':
        # drop Donor if they do not have Cell_type, Donor in all the Time_points
        adata_grouped = adata_subset_.obs[
            adata_subset_.obs.groupby(['Donor', 'Cell_type'])['Time_point'].transform(
                'nunique'
            )
            == 4
        ]
        dropped_donors = (
            adata_subset.obs['Donor'].nunique() - adata_grouped['Donor'].nunique()
        )
        print(f'dropped {dropped_donors} donors')
        resting_cells = adata_grouped.loc[adata_grouped['Time_point'] == '0h', :]
        grouped = adata_grouped.groupby(['Donor', 'Cell_type'])
        for idx, resting in tqdm.tqdm(
            resting_cells.iterrows(), total=resting_cells.shape[0]
        ):
            # get the indices of the other time points for the same cell type and donor
            group = grouped.get_group((resting['Donor'], resting['Cell_type']))
            indices_16h = group[group['Time_point'] == '16h'].index
            indices_40h = group[group['Time_point'] == '40h'].index
            indices_5d = group[group['Time_point'] == '5d'].index
            cell_pairings['0h'].append(idx)
            cell_pairings['16h'].append(np.random.choice(indices_16h))
            cell_pairings['40h'].append(np.random.choice(indices_40h))
            cell_pairings['5d'].append(np.random.choice(indices_5d))

    elif pairing_mode == 'random':
        # randomly sample from each time point
        ref_adata = adata_dict[reference_time]
        cell_pairings[reference_time] = ref_adata.index.tolist()
        # remove reference time from dictionary
        del adata_dict[reference_time]
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
