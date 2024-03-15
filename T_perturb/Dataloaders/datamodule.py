import pickle
from typing import Optional
from warnings import warn

import numpy as np

# import scanpy as sc
import torch
from datasets import DatasetDict
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset


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
        src_counts: Optional[np.ndarray] = None,
        tgt_counts: Optional[np.ndarray] = None,
        split_indices: Optional[list] = None,
        tgt_size_factor: Optional[np.ndarray] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[dict] = None,
    ):
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_datasets = tgt_datasets
        self.src_counts = src_counts
        self.tgt_counts = tgt_counts
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        self.condition_encodings = condition_encodings
        self.tgt_size_factor = tgt_size_factor
        if split_indices is None:
            self.src_dataset = src_dataset
            self.tgt_datasets = tgt_datasets
            self.src_counts = src_counts
            self.tgt_counts = tgt_counts
        else:
            self.src_dataset = src_dataset.select(split_indices)
            # self.tgt_dataset = tgt_dataset.select(split_indices)
            tmp_tgt_datasets = tgt_datasets.copy()
            for key, dataset in tmp_tgt_datasets.items():
                tmp_tgt_datasets[key] = dataset.select(split_indices)
            self.tgt_datasets = tmp_tgt_datasets

            if src_counts is not None:
                self.src_counts = src_counts[split_indices, :]
            if tgt_counts is not None:
                self.tgt_counts = tgt_counts[split_indices, :]

    def __getitem__(self, ind):
        return {
            'src_dataset': self.src_dataset[ind],
            # 'tgt_dataset': self.tgt_datasets[ind],
            'tgt_dataset_t1': self.tgt_datasets['t1'][ind],
            'tgt_dataset_t2': self.tgt_datasets['t2'][ind],
            'tgt_counts': self.tgt_counts[ind],
            'src_counts': self.src_counts[ind],
            'tgt_size_factor': self.tgt_size_factor[ind],
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
        split: bool = False,
        src_counts: Optional[np.ndarray] = None,
        tgt_counts: Optional[np.ndarray] = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        train_indices: Optional[list] = None,
        val_indices: Optional[list] = None,
        test_indices: Optional[list] = None,
        tgt_size_factor: Optional[np.ndarray] = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for Petra tokenised data.
        """
        super().__init__()
        print('Start datamodule')
        self.src_dataset = src_dataset
        self.tgt_datasets = tgt_datasets
        self.src_counts = src_counts
        self.tgt_counts = tgt_counts
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
        self.tgt_size_factor = tgt_size_factor

        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.condition_encodings is not None:
                self.train_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    tgt_size_factor=self.tgt_size_factor,
                    split_indices=self.train_indices,
                    src_counts=self.src_counts,
                    tgt_counts=self.tgt_counts,
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
                        tgt_size_factor=self.tgt_size_factor,
                        split_indices=self.val_indices,
                        src_counts=self.src_counts,
                        tgt_counts=self.tgt_counts,
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
                    tgt_size_factor=self.tgt_size_factor,
                    split_indices=self.train_indices,
                    src_counts=self.src_counts,
                    tgt_counts=self.tgt_counts,
                )
                if self.val_indices is not None:
                    self.val_dataset = PetraDataset(
                        src_dataset=self.src_dataset,
                        tgt_datasets=self.tgt_datasets,
                        tgt_size_factor=self.tgt_size_factor,
                        split_indices=self.val_indices,
                        src_counts=self.src_counts,
                        tgt_counts=self.tgt_counts,
                    )
                else:
                    self.val_dataset = None
        if stage == 'test' or stage is None:
            if self.condition_encodings is not None:
                self.test_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    tgt_size_factor=self.tgt_size_factor,
                    split_indices=self.test_indices,
                    src_counts=self.src_counts,
                    tgt_counts=self.tgt_counts,
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
                    tgt_size_factor=self.tgt_size_factor,
                    split_indices=self.test_indices,
                    src_counts=self.src_counts,
                    tgt_counts=self.tgt_counts,
                )

    def train_dataloader(self):
        data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
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
            # persistent_workers=True,
        )
        return data

    @staticmethod
    def return_tgt_dataset(
        dataset_name,
        batch,
        max_len,
        pad_token_id,
    ):
        if any(dataset_name in item for item in batch):
            tgt_input_batch_id = [
                torch.tensor(d[dataset_name]['input_ids'], device='cpu') for d in batch
            ]
            tgt_length = torch.stack(
                [torch.tensor(d[dataset_name]['length'], device='cpu') for d in batch]
            )
            model_input_size = torch.max(tgt_length)
            tgt_cell_population = [d[dataset_name]['Cell_population'] for d in batch]
            tgt_input_batch_id = pad_tensor_list(
                tgt_input_batch_id,
                max_len,
                pad_token_id,
                model_input_size,
            )
        else:
            tgt_input_batch_id = None
            tgt_length = None
            tgt_cell_population = None
        return tgt_input_batch_id, tgt_length, tgt_cell_population

    def collate(self, batch):
        if any('src_dataset' in item for item in batch):
            src_input_batch_id = [
                torch.tensor(d['src_dataset']['input_ids'], device='cpu') for d in batch
            ]
            src_length = torch.stack(
                [torch.tensor(d['src_dataset']['length'], device='cpu') for d in batch]
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

        (
            tgt_input_batch_id_t1,
            tgt_length_t1,
            tgt_cell_population_t1,
        ) = PetraDataModule.return_tgt_dataset(
            'tgt_dataset_t1',
            batch,
            self.max_len,
            self.pad_token_id,
        )
        (
            tgt_input_batch_id_t2,
            tgt_length_t2,
            tgt_cell_population_t2,
        ) = PetraDataModule.return_tgt_dataset(
            'tgt_dataset_t2',
            batch,
            self.max_len,
            self.pad_token_id,
        )

        if any('tgt_dataset_t1' in item for item in batch):
            tgt_cell_type = [d['tgt_dataset_t1']['Cell_type'] for d in batch]
            tgt_donor = [d['tgt_dataset_t1']['Donor'] for d in batch]
            tgt_time_point = [d['tgt_dataset_t1']['Time_point'] for d in batch]
        else:
            tgt_cell_type = None
            tgt_time_point = None
            tgt_donor = None

        if any('src_counts' in item for item in batch):
            src_counts = [torch.tensor(d['src_counts'], device='cpu') for d in batch]
            if isinstance(batch[0]['src_counts'], csr_matrix):
                src_counts = [torch.tensor(d['src_counts'].A) for d in batch]

            else:
                src_counts = [torch.tensor(d['src_counts']) for d in batch]
            src_counts = torch.cat(src_counts, dim=0)
        else:
            src_counts = None

        if any('tgt_counts' in item for item in batch):
            tgt_counts = [torch.tensor(d['tgt_counts'], device='cpu') for d in batch]
            if isinstance(batch[0]['tgt_counts'], csr_matrix):
                tgt_counts = [torch.tensor(d['tgt_counts'].A) for d in batch]
            else:
                tgt_counts = [torch.tensor(d['tgt_counts']) for d in batch]
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
            'tgt_size_factor': tgt_size_factor,
            'batch': condition,
            'combined_batch': condition_combined,
        }


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
