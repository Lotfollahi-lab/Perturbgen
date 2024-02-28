import pickle
from typing import Optional

import anndata as ad
import numpy as np

# import scanpy as sc
import torch
from datasets import DatasetDict
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    random_split,
)

from T_perturb.src.utils import label_encoder


# Dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, max_len, tgt_vocab_size):
        self.max_len = max_len
        self.tgt_vocab_size = tgt_vocab_size

    def __len__(self):
        return 1000  # Dummy number of samples

    def __getitem__(self, idx):
        # Dummy input data (replace with your actual data loading)
        src_input_ids = torch.randint(0, self.tgt_vocab_size, (self.max_len,))
        tgt_input_ids = torch.randint(0, self.tgt_vocab_size, (self.max_len,))
        src_input_ids[:, -5:] = 0
        tgt_input_ids[:, -5:] = 0

        return {
            'src': src_input_ids,
            'tgt': tgt_input_ids,
        }


class scConformerDataset(Dataset):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_dataset: DatasetDict,
        shuffle: bool = False,
        src_adata: ad.AnnData = None,
        tgt_adata: ad.AnnData = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[dict] = None,
    ):
        super().__init__()
        """
        Description:
        ------------
        This class load tokenised data from disk and extract the following information:
        - input_ids: tokenised gene expression data, padded to the same length
        - length: length of each cell
        """
        self.shuffle = shuffle
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.src_adata = src_adata
        self.tgt_adata = tgt_adata
        print('src_dataset', src_dataset)
        print('tgt_dataset', tgt_dataset)
        # check if the index is the same
        if not all(
            self.tgt_adata.obs['cell_pairing_index']
            == self.tgt_dataset['cell_pairing_index']
        ):
            raise ValueError('Index of adata and tokenized data do not match')

        self.size_factor = np.ravel(tgt_adata.X.sum(axis=1))
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        self.condition_encodings = condition_encodings

        # with open(token_dictionary_file, "rb") as f:
        #     self.gene_token_dict = pickle.load(f)
        # self.pad_token_id = self.gene_token_dict.get("<pad>")

    def __getitem__(self, ind):
        return {
            'src_dataset': self.src_dataset[ind],
            'tgt_dataset': self.tgt_dataset[ind],
            'tgt_adata': self.tgt_adata[ind],
            'src_adata': self.src_adata[ind],
            'tgt_size_factor': self.size_factor[ind],
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }

    def __len__(self):
        if len(self.src_dataset) != len(self.tgt_dataset):
            Warning('src and tgt dataset have different length')
        return min(len(self.src_dataset), len(self.tgt_dataset))
        # return self.num_samples


# two dataloader vs one dataloader
class scConformerDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_dataset: DatasetDict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        loss_mode: str = 'mse',
        src_adata: ad.AnnData = None,
        tgt_adata: ad.AnnData = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions_combined_encodings: Optional[dict] = None,
        drop_last: bool = False,
    ):
        """
        Description:
        ------------
        Custom datamodule for scConformer tokenised data.
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.src_adata = src_adata
        self.tgt_adata = tgt_adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = max_len
        self.dataset = None
        self.loss_mode = loss_mode
        self.size_factor = np.ravel(tgt_adata.X.sum(axis=1))
        self.condition_keys = condition_keys
        self.condition_encodings = condition_encodings
        self.conditions_combined_encodings = conditions_combined_encodings
        self.drop_last = drop_last
        # train test split
        self.train_prop = 0.8
        self.val_prop = 0.1
        self.test_prop = 0.1

        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys
        if (self.condition_encodings is not None) and (self.condition_keys is not None):
            self.conditions = [
                label_encoder(
                    tgt_adata,
                    encoder=self.condition_encodings[self.condition_keys[i]],
                    condition_key=self.condition_keys[i],
                )
                for i in range(len(self.condition_encodings))
            ]
            self.conditions = torch.tensor(self.conditions, dtype=torch.long).T
            self.conditions_combined = label_encoder(
                tgt_adata,
                encoder=self.conditions_combined_encodings,
                condition_key='conditions_combined',
            )
            self.conditions_combined = torch.tensor(
                self.conditions_combined, dtype=torch.long
            )

    def setup(self, stage=None):
        if self.condition_encodings is not None:
            self.dataset = scConformerDataset(
                src_dataset=self.src_dataset,
                tgt_dataset=self.tgt_dataset,
                src_adata=self.src_adata,
                tgt_adata=self.tgt_adata,
                shuffle=self.shuffle,
                conditions=self.conditions if self.condition_keys is not None else None,
                conditions_combined=self.conditions_combined
                if self.condition_keys is not None
                else None,
            )
        else:
            self.dataset = scConformerDataset(
                src_dataset=self.src_dataset,
                tgt_dataset=self.tgt_dataset,
                src_adata=self.src_adata,
                tgt_adata=self.tgt_adata,
                shuffle=self.shuffle,
            )
        # if self.train is None and self.val is None and self.test is None:
        #     self.train, self.val, self.test = self.train_test_val_split()

        # # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        #     self.train_dataset = self.train
        #     self.val_dataset = self.val

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.test_dataset = self.test

    def train_test_val_split(self):
        np.random.seed(self.seed)  # reproducibility
        if self.split == 'random':
            train, val, test = self.random_split()
        elif self.split == 'stratified':
            train, val, test = self.stratified_split()
        elif self.split == 'unseen_donor':
            train, val, test = self.unseen_donor_split()
        else:
            raise ValueError(
                "split is not available, must be either '"
                "random','stratified' or 'unseen_donor'"
            )
        print(
            f'Number of samples in train set: {len(train)}\n'
            f'Number of samples in val set: {len(val)}\n'
            f'Number of samples in test set: {len(test)}'
        )

        return train, val, test

    def train_dataloader(self):
        data = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=self.drop_last,
        )
        return data

    # def val_dataloader(self):
    #     data = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=self.shuffle,
    #         num_workers=self.num_workers,
    #         collate_fn=self.collate,
    #         drop_last=self.drop_last,
    #     )
    #     return data
    def test_dataloader(self):
        data = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=self.drop_last,
        )
        return data

    def collate(self, batch):
        if any('src_dataset' in item for item in batch):
            src_input_batch_id = [
                torch.tensor(d['src_dataset']['input_ids']) for d in batch
            ]
            src_length = torch.stack(
                [torch.tensor(d['src_dataset']['length']) for d in batch]
            )
            model_input_size = torch.max(src_length)
            src_cell_type = [d['src_dataset']['Cell_type'] for d in batch]
            src_time_point = ([d['src_dataset']['Time_point'] for d in batch],)
            src_donor = ([d['src_dataset']['Donor'] for d in batch],)
            src_input_batch_id = pad_tensor_list(
                src_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )
        else:
            src_input_batch_id = None
            src_length = None
            src_cell_type = None
            src_time_point = None
            src_donor = None

        if any('tgt_dataset' in item for item in batch):
            # return counts
            tgt_input_batch_id = [
                torch.tensor(d['tgt_dataset']['input_ids']) for d in batch
            ]
            tgt_length = torch.stack(
                [torch.tensor(d['tgt_dataset']['length']) for d in batch]
            )
            model_input_size = torch.max(tgt_length)
            tgt_cell_type = [d['tgt_dataset']['Cell_type'] for d in batch]
            tgt_cell_population = [d['tgt_dataset']['Cell_population'] for d in batch]
            tgt_time_point = ([d['tgt_dataset']['Time_point'] for d in batch],)
            tgt_donor = ([d['tgt_dataset']['Donor'] for d in batch],)
            tgt_input_batch_id = pad_tensor_list(
                tgt_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )
        else:
            tgt_input_batch_id = None
            tgt_length = None
            tgt_cell_type = None
            tgt_cell_population = None
            tgt_time_point = None
            tgt_donor = None

        if any('src_adata' in item for item in batch):
            src_counts = [torch.tensor(d['src_adata'].X) for d in batch]
            src_counts = torch.cat(src_counts, dim=0)
        else:
            src_counts = None

        if any('tgt_adata' in item for item in batch):
            tgt_counts = [torch.tensor(d['tgt_adata'].X) for d in batch]
            tgt_counts = torch.cat(tgt_counts, dim=0)
            tgt_size_factor = [d['tgt_size_factor'] for d in batch]
            if self.condition_encodings is not None:
                condition = [d['conditions'] for d in batch]
                condition_combined = [d['conditions_combined'] for d in batch]
            else:
                condition = None
                condition_combined = None
        else:
            tgt_counts = None
            tgt_size_factor = None

        return {
            'src_input_ids': src_input_batch_id,
            'src_length': src_length,
            'src_cell_type': src_cell_type,
            'src_time_point': src_time_point,
            'src_donor': src_donor,
            'src_counts': src_counts,
            'tgt_input_ids': tgt_input_batch_id,
            'tgt_length': tgt_length,
            'tgt_cell_type': tgt_cell_type,
            'tgt_cell_population': tgt_cell_population,
            'tgt_time_point': tgt_time_point,
            'tgt_donor': tgt_donor,
            'tgt_counts': tgt_counts,
            'size_factor': tgt_size_factor,
            'batch': condition,
            'combined_batch': condition_combined,
        }

    def gen_attention_mask(self, length):
        attention_mask = [
            [1] * original_len + [0] * (self.max_len - original_len)
            if original_len <= self.max_len
            else [1] * self.max_len
            for original_len in length
        ]

        return torch.tensor(attention_mask)
        # can change the function to make it more generic
        # -> only return train, val and test indices

    def random_split(self):
        # define train, val and test size
        train_size = np.round(self.train_prop * self.dataset.__len__()).astype(int)
        test_size = np.round(self.test_prop * self.dataset.__len__()).astype(int)
        val_size = self.dataset.__len__() - train_size - test_size
        generator = torch.Generator().manual_seed(self.seed)
        train, val, test = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )

        return train, val, test

    def stratified_split(self):
        # define groups for stratified split by Time_point and Cell_type
        groups = self.dataset.adata.obs['Cell_type'].copy()
        # combine to one column
        groups.loc[:, 'stratified'] = groups.loc[:, ['Cell_type']].apply(
            lambda x: '_'.join(x), axis=1
        )
        # define train, val and test size based on unique groups
        # extract unique groups and counts
        unique_groups = groups['stratified'].unique()
        group_indices = [np.where(groups == i)[0] for i in unique_groups]
        train_indices, test_indices, val_indices = [], [], []
        count = 0
        for indices in group_indices:
            assert (
                len(np.unique(groups.iloc[indices].stratified)) == 1
            ), 'groups are not stratified'
            # randomly shuffle indices
            np.random.shuffle(indices)
            train_size = np.round(self.train_prop * len(indices)).astype(int)
            test_size = np.round(self.test_prop * len(indices)).astype(int)
            # val_size = len(indices) - train_size - test_size

            # split indices into train, val and test set
            train_indices.extend(indices[:train_size])
            test_indices.extend(indices[train_size : train_size + test_size])
            val_indices.extend(indices[train_size + test_size :])
            count += 1

            # assert len(np.unique(groups.iloc[train_indices])) == 1
        train = Subset(self.dataset, train_indices)
        val = Subset(self.dataset, val_indices)
        test = Subset(self.dataset, test_indices)

        return train, val, test

    def unseen_donor_split(self):
        # define groups for stratified split by Time_point and Cell_type
        groups = self.dataset.adata.obs[['Donor']]
        # define train, val and test size based on unique donors
        train_size = np.round(self.train_prop * len(groups['Donor'].unique())).astype(
            int
        )
        test_size = np.round(self.test_prop * len(groups['Donor'].unique())).astype(int)
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
        train = Subset(self.dataset, np.where(groups['Donor'].isin(train_donors))[0])
        val = Subset(self.dataset, np.where(groups['Donor'].isin(val_donors))[0])
        test = Subset(self.dataset, np.where(groups['Donor'].isin(test_donors))[0])

        return train, val, test


if __name__ == '__main__':
    # test dataloader
    data_module = scConformerDataModule(
        src_dataset=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            'T_perturb/T_perturb/pp/res/dataset/'
            'cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset'
        ),
        tgt_dataset=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            'T_perturb/T_perturb/pp/res/dataset/'
            'cytoimmgen_tokenised_degs_stratified_pairing_16h.dataset'
        ),
        max_len=334,
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()
    # iterate through batches
    train_iterator = iter(dataloader)
    batch = next(train_iterator)
    print(batch['tgt_input_ids'][:20, :20])
    print(len(batch['tgt_counts'][0]))
