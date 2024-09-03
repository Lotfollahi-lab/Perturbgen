import pickle
from typing import (
    Dict,
    List,
    Optional,
)
from warnings import warn

import numpy as np

# import scanpy as sc
import torch
from datasets import DatasetDict
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pandas import Categorical
from pytorch_lightning import LightningDataModule
from scipy.sparse import csr_matrix
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
)

from T_perturb.src.utils import weighted_sampler


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


class CellGenDataset(Dataset):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_datasets: DatasetDict,
        split_indices: Dict = {},
        src_counts: np.ndarray = None,
        tgt_counts_dict: np.ndarray = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[Dict] = None,
    ):
        super().__init__()
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        self.condition_encodings = condition_encodings

        self.src_dataset_paired: DatasetDict = None
        self.tgt_dataset_paired: DatasetDict = None
        self.src_counts_paired: np.ndarray = None
        self.tgt_counts_paired: np.ndarray = None

        src_indices = list(split_indices.keys())
        self.src_dataset_paired = src_dataset.select(src_indices)
        if src_counts is not None:
            self.src_counts_paired = src_counts[src_indices, :]
        tgt_indices = list(split_indices.values())

        self.tgt_dataset_paired = tgt_datasets.select(tgt_indices)
        if tgt_counts_dict is not None:
            self.tgt_counts_paired = tgt_counts_dict[tgt_indices, :]
        src_len = len(self.src_dataset_paired)
        tgt_len = len(self.tgt_dataset_paired)
        if src_len != tgt_len:
            warn('src and tgt dataset have different length')
        self.dataset_length = min(src_len, tgt_len)

    def __getitem__(self, ind):
        out = {
            'src_dataset': self.src_dataset_paired[ind],
            'src_counts': self.src_counts_paired[ind]
            if self.src_counts_paired is not None
            else None,
            'tgt_dataset': self.tgt_dataset_paired[ind],
            'tgt_counts': self.tgt_counts_paired[ind]
            if self.tgt_counts_paired is not None
            else None,
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }
        return out

    def __len__(self):
        return self.dataset_length


# two dataloader vs one dataloader
class CellGenDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_datasets: DatasetDict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        split: bool = False,
        src_counts: Optional[np.ndarray] = None,
        tgt_counts_dict: Optional[np.ndarray] = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
        test_indices: Optional[list[int]] = None,
        train_dict: Optional[Dict] = None,
        val_dict: Optional[Dict] = None,
        test_dict: Optional[Dict] = None,
        var_list: Optional[list] = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for CellGen tokenised data.
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_datasets = tgt_datasets
        self.src_counts = src_counts
        self.tgt_counts_dict = tgt_counts_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = max_len
        self.dataset = None
        self.condition_keys = condition_keys
        self.condition_encodings = condition_encodings
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        # train test split
        self.split = split
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        self.var_list = var_list

        self.train_dataset_list: List[Dataset] = []
        self.val_dataset_list: List[Dataset] = []
        self.test_dataset_list: List[Dataset] = []
        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys

    def setup(self, stage=None):
        for i in range(1, len(self.tgt_datasets) + 1):
            str_i = str(i)
            dataset_params = {
                'src_dataset': self.src_dataset,
                'tgt_datasets': self.tgt_datasets[f'tgt_dataset_t{str_i}'],
                'tgt_counts_dict': self.tgt_counts_dict[f'tgt_h5ad_t{str_i}']
                if self.tgt_counts_dict is not None
                else None,
                'src_counts': self.src_counts if self.src_counts is not None else None,
            }

            if self.condition_encodings is not None:
                condition_params = {
                    'conditions': self.conditions
                    if self.condition_keys is not None
                    else None,
                    'conditions_combined': self.conditions_combined
                    if self.condition_keys is not None
                    else None,
                }
                dataset_params.update(condition_params)

            mapping_dict_key = f'tgt_pkl_t{str_i}'
            if stage == 'fit' or stage is None:
                train_dataset = CellGenDataset(
                    split_indices=self.train_dict[mapping_dict_key], **dataset_params
                )
                self.train_dataset_list.append(train_dataset)
                if self.val_dict is not None:
                    self.val_dataset = CellGenDataset(
                        split_indices=self.val_dict[mapping_dict_key], **dataset_params
                    )
                    self.val_dataset_list.append(self.val_dataset)
                else:
                    self.val_dataset_list = None

            if stage == 'test' or stage is None:
                test_dataset = CellGenDataset(
                    split_indices=self.test_dict[mapping_dict_key], **dataset_params
                )
                self.test_dataset_list.append(test_dataset)

    def train_dataloader(self):
        train_dataset = ConcatDataset(self.train_dataset_list)
        train_sampler = weighted_sampler(self.train_dataset_list)

        data = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            sampler=train_sampler,
        )
        return data

    def val_dataloader(self):
        if self.split:
            if self.val_dict is not None:
                val_dataset = ConcatDataset(self.val_dataset_list)
                data = DataLoader(
                    dataset=val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=self.collate,
                )
                return data
            else:
                return []
        else:
            return []

    def test_dataloader(self):
        test_dataset = ConcatDataset(self.test_dataset_list)
        data = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return data

    def collate(self, batch):
        # src
        src_input_ids = [torch.tensor(d['src_dataset']['input_ids']) for d in batch]
        src_length = torch.tensor([d['src_dataset']['length'] for d in batch])
        model_input_size = torch.max(src_length)
        src_input_ids = pad_tensor_list(
            src_input_ids, self.max_len, self.pad_token_id, model_input_size
        )

        if batch[0]['src_counts'] is not None:
            if isinstance(batch[0]['src_counts'], csr_matrix):
                src_counts = [torch.tensor(d['src_counts'].A) for d in batch]
            else:
                src_counts = [torch.tensor(d['src_counts']) for d in batch]
            src_counts = torch.cat(src_counts, dim=0)
        else:
            src_counts = None

        if self.condition_encodings:
            condition = [d['conditions'] for d in batch]
            condition_combined = torch.stack([d['conditions_combined'] for d in batch])
        else:
            condition, condition_combined = None, None

        # tgt
        tgt_input_ids = [torch.tensor(d['tgt_dataset']['input_ids']) for d in batch]
        tgt_length = torch.tensor([d['tgt_dataset']['length'] for d in batch])
        tgt_input_ids = pad_tensor_list(
            tgt_input_ids, self.max_len, self.pad_token_id, model_input_size
        )
        model_input_size = torch.max(tgt_length)
        if batch[0]['tgt_counts'] is not None:
            if isinstance(batch[0]['tgt_counts'], csr_matrix):
                tgt_counts_matrices = [d['tgt_counts'].A for d in batch]
            else:
                tgt_counts_matrices = [d['tgt_counts'] for d in batch]

            tgt_counts = [torch.tensor(matrix) for matrix in tgt_counts_matrices]
            tgt_size_factor = [
                torch.tensor(np.ravel(matrix.sum(axis=1)))
                for matrix in tgt_counts_matrices
            ]
            tgt_counts = torch.cat(tgt_counts, dim=0)
            tgt_size_factor = torch.cat(tgt_size_factor, dim=0)
        else:
            tgt_counts = None
            tgt_size_factor = None
        tgt_cell_idx = [d['tgt_dataset']['cell_pairing_index'] for d in batch]
        out = {
            'src_input_ids': src_input_ids,
            'src_length': src_length,
            'src_counts': src_counts,
            'batch': condition,
            'combined_batch': condition_combined,
            'tgt_input_ids': tgt_input_ids,
            'tgt_length': tgt_length,
            'tgt_counts': tgt_counts,
            'tgt_size_factor': tgt_size_factor,
            'tgt_cell_idx': tgt_cell_idx,
        }
        for var in self.var_list:
            out[var] = [d['tgt_dataset'][var] for d in batch]
        if out['disease'] is not None:
            categories = out['disease']
            num_classes = len(set(categories))
            # Convert categories to indices and then to one-hot encoding
            category_indices = torch.tensor(
                Categorical(categories).codes, dtype=torch.long
            )
            one_hot_labels = torch.eye(num_classes)[category_indices]
            out['moe_categories'] = one_hot_labels
            out['moe_num_classes'] = num_classes
        return out


if __name__ == '__main__':
    # test dataloader
    data_module = CellGenDataModule(
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
