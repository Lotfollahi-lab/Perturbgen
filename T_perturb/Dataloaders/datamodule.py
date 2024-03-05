import pickle
from typing import Optional
from warnings import warn

import anndata as ad
import numpy as np

# import scanpy as sc
import torch
from datasets import DatasetDict
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from T_perturb.src.utils import label_encoder, stratified_split


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


class PetraDataset(Dataset):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_datasets: DatasetDict,
        src_adata: ad.AnnData,
        tgt_adata: ad.AnnData,
        shuffle: bool = False,
        split_indices: Optional[list] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[dict] = None,
    ):
        super().__init__()
        self.shuffle = shuffle
        if split_indices is None:
            self.src_dataset = src_dataset
            self.tgt_datasets = tgt_datasets
            self.src_adata = src_adata
            self.tgt_adata = tgt_adata
        else:
            self.src_dataset = src_dataset.select(split_indices)
            # self.tgt_dataset = tgt_dataset.select(split_indices)
            for key, dataset in tgt_datasets.items():
                tgt_datasets[key] = dataset.select(split_indices)
            self.tgt_datasets = tgt_datasets

            if src_adata is not None:
                self.src_adata = src_adata[split_indices, :]
            if tgt_adata is not None:
                self.tgt_adata = tgt_adata[split_indices, :]
        if tgt_adata is not None:
            self.size_factor = np.ravel(self.tgt_adata.X.sum(axis=1))
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        self.condition_encodings = condition_encodings

    def __getitem__(self, ind):
        return {
            'src_dataset': self.src_dataset[ind],
            # 'tgt_dataset': self.tgt_datasets[ind],
            'tgt_dataset_t1': self.tgt_datasets['t1'][ind],
            'tgt_dataset_t2': self.tgt_datasets['t2'][ind],
            'tgt_adata': self.tgt_adata[ind],
            'src_adata': self.src_adata[ind],
            'tgt_size_factor': self.size_factor[ind],
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }

    def __len__(self):
        if len(self.src_dataset) != len(self.tgt_datasets['t1']):
            warn('src and tgt dataset have different length')
        return min(len(self.src_dataset), len(self.tgt_datasets['t1']))


# two dataloader vs one dataloader
class PetraDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_datasets: DatasetDict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        seed: int = 42,
        splitting_mode: str = 'stratified',  # 'random', 'stratified', 'unseen_donor
        src_adata: ad.AnnData = None,
        tgt_adata: ad.AnnData = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions_combined_encodings: Optional[dict] = None,
        drop_last: bool = False,
        split: bool = False,
    ):
        """
        Description:
        ------------
        Custom datamodule for Petra tokenised data.
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_datasets = tgt_datasets
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
        self.size_factor = np.ravel(tgt_adata.X.sum(axis=1))
        self.condition_keys = condition_keys
        self.condition_encodings = condition_encodings
        self.conditions_combined_encodings = conditions_combined_encodings
        self.drop_last = drop_last
        # train test split
        self.split = split
        self.splitting_mode = splitting_mode
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_prop = 0.8
        self.val_prop = 0.1
        self.test_prop = 0.1
        self.seed = seed

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
        if self.split:
            if (
                self.train_indices is None
                and self.val_indices is None
                and self.test_indices is None
            ):
                (
                    self.train_indices,
                    self.val_indices,
                    self.test_indices,
                ) = self.train_test_val_split()
            else:
                # return all the indices
                self.train_indices = list(range(len(self.src_dataset)))
                self.val_indices = None
                self.test_indices = list(range(len(self.tgt_datasets)))

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.condition_encodings is not None:
                self.train_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    split_indices=self.train_indices,
                    src_adata=self.src_adata,
                    tgt_adata=self.tgt_adata,
                    shuffle=self.shuffle,
                    conditions=self.conditions
                    if self.condition_keys is not None
                    else None,
                    conditions_combined=self.conditions_combined
                    if self.condition_keys is not None
                    else None,
                )
                if self.val_indices is not None:
                    self.val_dataset = PetraDataset(
                        src_dataset=self.src_dataset,
                        tgt_datasets=self.tgt_datasets,
                        split_indices=self.val_indices,
                        src_adata=self.src_adata,
                        tgt_adata=self.tgt_adata,
                        shuffle=self.shuffle,
                        conditions=self.conditions
                        if self.condition_keys is not None
                        else None,
                        conditions_combined=self.conditions_combined
                        if self.condition_keys is not None
                        else None,
                    )
                else:
                    self.val_dataset = None
            else:
                self.train_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    split_indices=self.train_indices,
                    src_adata=self.src_adata,
                    tgt_adata=self.tgt_adata,
                    shuffle=self.shuffle,
                )
                if self.val_indices is not None:
                    self.val_dataset = PetraDataset(
                        src_dataset=self.src_dataset,
                        tgt_datasets=self.tgt_datasets,
                        split_indices=self.val_indices,
                        src_adata=self.src_adata,
                        tgt_adata=self.tgt_adata,
                        shuffle=self.shuffle,
                    )
                else:
                    self.val_dataset = None
        if stage == 'test' or stage is None:
            if self.condition_encodings is not None:
                self.test_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    split_indices=self.test_indices,
                    src_adata=self.src_adata,
                    tgt_adata=self.tgt_adata,
                    shuffle=self.shuffle,
                    conditions=self.conditions
                    if self.condition_keys is not None
                    else None,
                    conditions_combined=self.conditions_combined
                    if self.condition_keys is not None
                    else None,
                )
            else:
                self.test_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    split_indices=self.test_indices,
                    src_adata=self.src_adata,
                    tgt_adata=self.tgt_adata,
                    shuffle=self.shuffle,
                )

    def train_test_val_split(self):
        np.random.seed(self.seed)  # reproducibility

        if self.splitting_mode == 'stratified':
            train_indices, val_indices, test_indices = stratified_split(
                self.tgt_adata,
                self.train_prop,
                self.test_prop,
                ['Cell_type', 'Donor'],
                self.seed,
            )
            # check that indices are unique to avoid data leakage
            assert len(set(train_indices).intersection(val_indices)) == 0
            assert len(set(train_indices).intersection(test_indices)) == 0
            assert len(set(val_indices).intersection(test_indices)) == 0
        # elif self.split == 'random':
        #     train, val, test = self.random_split()
        # elif self.split == 'unseen_donor':
        #     train, val, test = self.unseen_donor_split()
        else:
            raise ValueError(
                "split is not available, must be either '"
                "random','stratified' or 'unseen_donor'"
            )
        print(
            f'Number of samples in train set: {len(train_indices)}\n'
            f'Number of samples in val set: {len(val_indices)}\n'
            f'Number of samples in test set: {len(test_indices)}'
        )

        return train_indices, val_indices, test_indices

    def train_dataloader(self):
        data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last=self.drop_last,
        )
        return data

    def val_dataloader(self):
        if self.split:
            data = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate,
                drop_last=self.drop_last,
            )
            return data
        else:
            return []

    def test_dataloader(self):
        data = DataLoader(
            self.test_dataset,
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

        @staticmethod
        def return_tgt_dataset(dataset_name):
            if any(dataset_name in item for item in batch):
                tgt_input_batch_id = [
                    torch.tensor(d[dataset_name]['input_ids']) for d in batch
                ]
                tgt_length = torch.stack(
                    [torch.tensor(d[dataset_name]['length']) for d in batch]
                )
                model_input_size = torch.max(tgt_length)
                tgt_cell_population = [
                    d[dataset_name]['Cell_population'] for d in batch
                ]
                tgt_input_batch_id = pad_tensor_list(
                    tgt_input_batch_id,
                    self.max_len,
                    self.pad_token_id,
                    model_input_size,
                )
            else:
                tgt_input_batch_id = None
                tgt_length = None
                tgt_cell_population = None
            return tgt_input_batch_id, tgt_length, tgt_cell_population

        (
            tgt_input_batch_id_t1,
            tgt_length_t1,
            tgt_cell_population_t1,
        ) = return_tgt_dataset('tgt_dataset_t1')
        (
            tgt_input_batch_id_t2,
            tgt_length_t2,
            tgt_cell_population_t2,
        ) = return_tgt_dataset('tgt_dataset_t2')

        if any('tgt_dataset_t1' in item for item in batch):
            tgt_cell_type = [d['tgt_dataset_t1']['Cell_type'] for d in batch]
            tgt_donor = [d['tgt_dataset_t1']['Donor'] for d in batch]
            tgt_time_point = [d['tgt_dataset_t1']['Time_point'] for d in batch]
        else:
            tgt_cell_type = None
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
            'tgt_input_ids_t1': tgt_input_batch_id_t1,
            'tgt_input_ids_t2': tgt_input_batch_id_t2,
            'tgt_length_t1': tgt_length_t1,
            'tgt_length_t2': tgt_length_t2,
            'tgt_cell_population_t1': tgt_cell_population_t1,
            'tgt_cell_population_t2': tgt_cell_population_t2,
            'tgt_cell_type': tgt_cell_type,
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


if __name__ == '__main__':
    # test dataloader
    data_module = PetraDataModule(
        src_dataset=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            'T_perturb/T_perturb/pp/res/dataset/'
            'cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset'
        ),
        tgt_datasets=(
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
