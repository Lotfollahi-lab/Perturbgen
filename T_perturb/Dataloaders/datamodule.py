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

from T_perturb.src.utils import label_encoder, stratified_split, randomised_split, gears_splitter


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
        tgt_dataset: DatasetDict,
        src_counts: np.ndarray = None,
        tgt_counts: np.ndarray = None,
        split_indices: Optional[list] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[dict] = None,
    ):
        super().__init__()
        if split_indices is None:
            self.src_dataset = src_dataset
            self.tgt_dataset = tgt_dataset
            self.src_counts = src_counts
            self.tgt_counts = tgt_counts
        else:
            self.src_dataset = src_dataset.select(split_indices)
            self.tgt_dataset = tgt_dataset.select(split_indices)
            self.src_counts = src_counts[split_indices, :]
            self.tgt_counts = tgt_counts[split_indices, :]

        if tgt_counts is not None:
            self.size_factor = torch.tensor(np.ravel(self.tgt_counts.sum(axis=1)))
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        self.condition_encodings = condition_encodings

    def __getitem__(self, ind):
        return {
            'src_dataset': self.src_dataset[ind],
            'tgt_dataset': self.tgt_dataset[ind],
            'tgt_counts': self.tgt_counts[ind, :],
            'src_counts': self.src_counts[ind],
            'tgt_size_factor': self.size_factor[ind],
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind] if self.conditions_combined is not None else None,
        }

    def __len__(self):
        if len(self.src_dataset) != len(self.tgt_dataset):
            warn('src and tgt dataset have different length')
        return min(len(self.src_dataset), len(self.tgt_dataset))

# two dataloader vs one dataloader
class PetraDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_dataset: DatasetDict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        src_counts: Optional[np.ndarray] = None,
        tgt_counts: Optional[np.ndarray] = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        drop_last: bool = False,
        split: bool = False,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
        test_indices: Optional[list[int]] = None,       
    ):
        """
        Description:
        ------------
        Custom datamodule for Petra tokenised data.
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
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
        self.drop_last = drop_last
        # train test split
        self.split = split
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.condition_encodings is not None:
                self.train_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_dataset=self.tgt_dataset,
                    split_indices=self.train_indices,
                    src_counts=self.src_counts,
                    tgt_counts=self.tgt_counts,
                    conditions=self.conditions if self.condition_keys is not None else None,
                    conditions_combined=self.conditions_combined if self.condition_keys is not None else None,
                )
                if self.val_indices is not None:
                    self.val_dataset = PetraDataset(
                        src_dataset=self.src_dataset,
                        tgt_dataset=self.tgt_dataset,
                        split_indices=self.val_indices,
                        src_counts=self.src_counts,
                        tgt_counts=self.tgt_counts,
                        conditions=self.conditions if self.condition_keys is not None else None,
                        conditions_combined=self.conditions_combined if self.condition_keys is not None else None,
                    )
                else:
                    self.val_dataset = None
            else:
                self.train_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_dataset=self.tgt_dataset,
                    split_indices=self.train_indices,
                    src_counts=self.src_counts,
                    tgt_counts=self.tgt_counts,
                )
                if self.val_indices is not None:
                    self.val_dataset = PetraDataset(
                        src_dataset=self.src_dataset,
                        tgt_dataset=self.tgt_dataset,
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
                    tgt_dataset=self.tgt_dataset,
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
                    tgt_dataset=self.tgt_dataset,
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
            src_input_batch_id = pad_tensor_list(
                src_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )

            # To do: make more general (key parameter)
            if any('Cell_type' in item['src_dataset'].keys() for item in batch):
                src_cell_type = [d['src_dataset']['Cell_type'] for d in batch]
                src_time_point = ([d['src_dataset']['Time_point'] for d in batch],)
                src_donor = ([d['src_dataset']['Donor'] for d in batch],)
            else:
                src_cell_type = None
                src_time_point = None
                src_donor = None

            if any('perturbation_id' in item['src_dataset'].keys() for item in batch):
                perturbation_id = [d['src_dataset']['perturbation_id'] for d in batch]
                perturbation_embedding = torch.stack([torch.stack([torch.tensor(embed) for embed in d['src_dataset']['perturbation_embedding']]) for d in batch])

            else:
                perturbation_id = None
                perturbation_embedding = None

            src_cell_idx = [d['src_dataset']['cell_pairing_index'] for d in batch]

        else:
            src_input_batch_id = None
            src_length = None
            src_cell_type = None
            src_time_point = None
            src_donor = None
            perturbation_id = None
            perturbation_embedding = None
            src_cell_idx = None

        if any('tgt_dataset' in item for item in batch):
            tgt_input_batch_id = [
                torch.tensor(d['tgt_dataset']['input_ids']) for d in batch
            ]
            tgt_length = torch.stack(
                [torch.tensor(d['tgt_dataset']['length']) for d in batch]
            )
            model_input_size = torch.max(tgt_length)
            tgt_input_batch_id = pad_tensor_list(
                tgt_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )

            # To do: make more general (key parameter)
            if any('Cell_type' in item['tgt_dataset'].keys() for item in batch):
                tgt_cell_type = [d['tgt_dataset']['Cell_type'] for d in batch]
                tgt_cell_population = [d['tgt_dataset']['Cell_population'] for d in batch]
                tgt_time_point = ([d['tgt_dataset']['Time_point'] for d in batch],) # Bug: should be Cell Population in previous code
                tgt_donor = ([d['tgt_dataset']['Donor'] for d in batch],)
            else:
                tgt_cell_type = None
                tgt_cell_population = None
                tgt_time_point = None
                tgt_donor = None
            
            tgt_cell_idx = [d['tgt_dataset']['cell_pairing_index'] for d in batch]

        else:
            tgt_input_batch_id = None
            tgt_length = None
            tgt_cell_type = None
            tgt_cell_population = None
            tgt_time_point = None
            tgt_donor = None
            tgt_cell_idx = None

        if any('src_counts' in item for item in batch):
            src_counts = [torch.tensor(d['src_counts']) for d in batch]
            src_counts = torch.stack(src_counts)
        else:
            src_counts = None

        if any('tgt_counts' in item for item in batch):
            tgt_counts = [torch.tensor(d['tgt_counts']) for d in batch]
            tgt_counts = torch.stack(tgt_counts)
            tgt_size_factor = [d['tgt_size_factor'] for d in batch]
            if any('testing_genes_subset' in item['tgt_dataset'].keys() for item in batch):
                testing_genes_subset = [d['tgt_dataset']['testing_genes_subset'] for d in batch]
            else:
                testing_genes_subset = None
        else:
            tgt_counts = None
            tgt_size_factor = None
            testing_genes_subset = None

        if self.condition_encodings is not None:
            condition = [d['conditions'] for d in batch]
            condition_combined = [d['conditions_combined'] for d in batch]
            condition_combined = torch.stack(condition_combined)
        else:
            condition = None
            condition_combined = None

        return {
            'src_input_ids': src_input_batch_id,
            'src_counts': src_counts,
            'src_length': src_length,
            'src_cell_type': src_cell_type,
            'src_time_point': src_time_point,
            'src_donor': src_donor,
            'tgt_input_ids': tgt_input_batch_id,
            'tgt_counts': tgt_counts,
            'tgt_length': tgt_length,
            'tgt_cell_type': tgt_cell_type,
            'tgt_cell_population': tgt_cell_population,
            'tgt_time_point': tgt_time_point,
            'tgt_donor': tgt_donor,
            'size_factor': tgt_size_factor,
            'batch': condition,
            'combined_batch': condition_combined,
            'perturbation_id': perturbation_id,
            'perturbation_embedding': perturbation_embedding,
            'testing_genes_subset': testing_genes_subset,
            'src_cell_idx': src_cell_idx, 
            'tgt_cell_idx': tgt_cell_idx, 
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
