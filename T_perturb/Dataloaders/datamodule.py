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
        time_steps: list = [1, 2],
        src_counts: np.ndarray = None,
        tgt_counts_dict: np.ndarray = None,
        split_indices: Optional[list] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[dict] = None,
    ):
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_datasets = tgt_datasets
        self.src_counts = src_counts
        self.tgt_counts_dict = tgt_counts_dict
        self.conditions = conditions
        self.conditions_combined = conditions_combined
        self.condition_encodings = condition_encodings
        self.time_steps = time_steps
        if max(time_steps) > len(tgt_datasets):
            raise ValueError('Number of time steps is greater than number of datasets')
        if split_indices is None:
            self.src_dataset = src_dataset
            self.tgt_datasets = tgt_datasets
            self.src_counts = src_counts
            self.tgt_counts_dict = tgt_counts_dict
        else:
            self.src_dataset = src_dataset.select(split_indices)
            tmp_tgt_datasets = tgt_datasets.copy()
            tmp_tgt_counts_dict = tgt_counts_dict.copy()
            self.time_steps = time_steps
            for t in self.time_steps:
                dataset = tmp_tgt_datasets[f'tgt_dataset_t{t}']
                tmp_tgt_datasets[f'tgt_dataset_t{t}'] = dataset.select(split_indices)
                if tgt_counts_dict is not None:
                    # time point split
                    counts = tmp_tgt_counts_dict[f'tgt_h5ad_t{t}']
                    tmp_tgt_counts_dict[f'tgt_h5ad_t{t}'] = counts[split_indices, :]
            self.tgt_datasets = tmp_tgt_datasets
            self.tgt_counts_dict = tmp_tgt_counts_dict

            if src_counts is not None:
                self.src_counts = src_counts[split_indices, :]

    def __getitem__(self, ind):
        out = {
            'src_dataset': self.src_dataset[ind],
            'src_counts': self.src_counts[ind],
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }
        for t in self.time_steps:
            dataset_keys_ = f'tgt_dataset_t{t}'

            out[dataset_keys_] = self.tgt_datasets[dataset_keys_][ind]
            out[f'tgt_counts_t{t}'] = self.tgt_counts_dict[f'tgt_h5ad_t{t}'][ind]

        return out

    def __len__(self):
        if len(self.src_dataset) != len(
            self.tgt_datasets[f'tgt_dataset_t{self.time_steps[0]}']
        ):
            warn('src and tgt dataset have different length')
        return min(
            len(self.src_dataset),
            len(self.tgt_datasets[f'tgt_dataset_t{self.time_steps[0]}']),
        )


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
        time_steps: list = [1, 2],
        src_counts: Optional[np.ndarray] = None,
        tgt_counts_dict: Optional[np.ndarray] = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
        test_indices: Optional[list[int]] = None,
        var_list: Optional[list] = None,
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
        self.time_steps = time_steps
        self.var_list = var_list
        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.condition_encodings is not None:
                self.train_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    split_indices=self.train_indices,
                    src_counts=self.src_counts,
                    tgt_counts_dict=self.tgt_counts_dict,
                    time_steps=self.time_steps,
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
                        src_counts=self.src_counts,
                        tgt_counts_dict=self.tgt_counts_dict,
                        time_steps=self.time_steps,
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
                    src_counts=self.src_counts,
                    tgt_counts_dict=self.tgt_counts_dict,
                    time_steps=self.time_steps,
                )
                if self.val_indices is not None:
                    self.val_dataset = PetraDataset(
                        src_dataset=self.src_dataset,
                        tgt_datasets=self.tgt_datasets,
                        split_indices=self.val_indices,
                        src_counts=self.src_counts,
                        tgt_counts_dict=self.tgt_counts_dict,
                        time_steps=self.time_steps,
                    )
                else:
                    self.val_dataset = None
        if stage == 'test' or stage is None:
            if self.condition_encodings is not None:
                self.test_dataset = PetraDataset(
                    src_dataset=self.src_dataset,
                    tgt_datasets=self.tgt_datasets,
                    split_indices=self.test_indices,
                    src_counts=self.src_counts,
                    tgt_counts_dict=self.tgt_counts_dict,
                    time_steps=self.time_steps,
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
                    src_counts=self.src_counts,
                    tgt_counts_dict=self.tgt_counts_dict,
                    time_steps=self.time_steps,
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
        else:
            src_input_batch_id = None
            src_length = None

        if any('src_counts' in item for item in batch):
            if isinstance(batch[0]['src_counts'], csr_matrix):
                src_counts = [torch.tensor(d['src_counts'].A) for d in batch]

            else:
                src_counts = [torch.tensor(d['src_counts']) for d in batch]
            src_counts = torch.cat(src_counts, dim=0)
        else:
            src_counts = None

        if self.condition_encodings is not None:
            condition = [d['conditions'] for d in batch]
            condition_combined = [d['conditions_combined'] for d in batch]
            condition_combined = torch.stack(condition_combined)
        else:
            condition = None
            condition_combined = None
        # compute tgt size factor
        out = {
            'src_input_ids': src_input_batch_id,
            'src_length': src_length,
            'src_counts': src_counts,
            'batch': condition,
            'combined_batch': condition_combined,
        }

        for time_step in self.time_steps:
            if isinstance(batch[0][f'tgt_counts_t{time_step}'], csr_matrix):
                tgt_counts = [
                    torch.tensor(d[f'tgt_counts_t{time_step}'].A) for d in batch
                ]
                tgt_size_factor = [
                    torch.tensor(np.ravel(d[f'tgt_counts_t{time_step}'].A.sum(axis=1)))
                    for d in batch
                ]
            else:
                tgt_counts = [
                    torch.tensor(d[f'tgt_counts_t{time_step}']) for d in batch
                ]
                tgt_size_factor = [
                    torch.tensor(np.ravel(d[f'tgt_counts_t{time_step}'].sum(axis=1)))
                    for d in batch
                ]
            out[f'tgt_counts_t{time_step}'] = torch.cat(tgt_counts, dim=0)
            out[f'tgt_size_factor_t{time_step}'] = torch.cat(tgt_size_factor, dim=0)

        for time_step in self.time_steps:
            dataset = f'tgt_dataset_t{time_step}'
            out[f'tgt_input_ids_t{time_step}'] = [
                torch.tensor(d[dataset]['input_ids'], device='cpu') for d in batch
            ]
            out[f'tgt_length_t{time_step}'] = torch.stack(
                [torch.tensor(d[dataset]['length'], device='cpu') for d in batch]
            )
            model_input_size = torch.max(out[f'tgt_length_t{time_step}'])
            for var in self.var_list:
                # if var == 'Time_point':
                #     time_step_list = [d[dataset]['Time_point'] for d in batch]
                #     out[f'{var}_t{time_step}'] = time_step_list
                #     # encode time point to categories for classification
                #     # encoder = LabelEncoder()
                #     # integer_time_step = encoder.fit_transform(time_step_list)
                #     # out[f'{var}_int_t{time_step}'] = torch.tensor(
                #     #     integer_time_step
                #     # )
                # else:
                out[f'{var}_t{time_step}'] = [d[dataset][var] for d in batch]

            out[f'tgt_input_ids_t{time_step}'] = pad_tensor_list(
                out[f'tgt_input_ids_t{time_step}'],
                self.max_len,
                self.pad_token_id,
                model_input_size,
            )

        return out


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
