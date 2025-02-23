import pickle
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


class CytoMeisterDataset(Dataset):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_datasets: DatasetDict,
        time_steps: list = [1, 2],
        src_counts: np.ndarray | None = None,
        tgt_counts_dict: np.ndarray | None = None,
        split_indices: list | None = None,
        conditions: torch.Tensor | None = None,
        conditions_combined: torch.Tensor | None = None,
        condition_encodings: dict | None = None,
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

        if split_indices is not None:
            self.src_dataset = src_dataset.select(split_indices)
            self.tgt_datasets = {}
            self.tgt_counts_dict = {}
            if self.src_counts is not None:
                self.src_counts = self.src_counts[split_indices, :]

            for t in time_steps:
                dataset_keys_ = f'tgt_dataset_t{t}'
                count_keys_ = f'tgt_h5ad_t{t}'
                self.tgt_datasets[dataset_keys_] = tgt_datasets[dataset_keys_].select(
                    split_indices
                )
                if tgt_counts_dict is not None:
                    self.tgt_counts_dict[count_keys_] = tgt_counts_dict[count_keys_][
                        split_indices, :
                    ]
        if max(time_steps) > len(tgt_datasets):
            raise ValueError('Number of time steps is greater than number of datasets')
        src_len = len(self.src_dataset)
        tgt_len = len(self.tgt_datasets[f'tgt_dataset_t{time_steps[0]}'])
        if src_len != tgt_len:
            warn('src and tgt dataset have different length')
        self.dataset_length = min(src_len, tgt_len)

    def __getitem__(self, ind):
        out = {
            'src_dataset': self.src_dataset[ind],
            'src_counts': self.src_counts[ind] if self.src_counts is not None else None,
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }
        for t in self.time_steps:
            dataset_keys_ = f'tgt_dataset_t{t}'
            out[dataset_keys_] = self.tgt_datasets[dataset_keys_][ind]
            if (self.tgt_counts_dict is not None) and (
                f'tgt_h5ad_t{t}' in self.tgt_counts_dict
            ):
                out[f'tgt_counts_t{t}'] = self.tgt_counts_dict[f'tgt_h5ad_t{t}'][ind]
            else:
                out[f'tgt_counts_t{t}'] = None
        return out

    def __len__(self):
        return self.dataset_length


class CytoMeisterDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_datasets: DatasetDict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        split: bool = False,
        pred_tps: list = [1, 2],
        n_total_tps: int = 4,
        src_counts: np.ndarray | None = None,
        tgt_counts_dict: np.ndarray | None = None,
        condition_keys: list | None = None,
        condition_encodings: dict | None = None,
        conditions: torch.Tensor | None = None,
        conditions_combined: torch.Tensor | None = None,
        train_indices: list[int] | None = None,
        val_indices: list[int] | None = None,
        test_indices: list[int] | None = None,
        var_list: list | None = None,
        context_tps: list | None = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for CytoMeister tokenised data.
        """
        super().__init__()
        print('Start datamodule')
        self.src_dataset = src_dataset
        self.tgt_datasets = tgt_datasets
        self.src_counts = src_counts
        self.tgt_counts_dict = tgt_counts_dict
        self.dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
        }
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
        self.pred_tps = pred_tps
        self.context_tps = context_tps
        self.total_tps = list(range(1, n_total_tps + 1))
        self.var_list = var_list
        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys

    def setup(self, stage=None):
        dataset_args = {
            'src_dataset': self.src_dataset,
            'tgt_datasets': self.tgt_datasets,
            'src_counts': self.src_counts,
            'tgt_counts_dict': self.tgt_counts_dict,
        }
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.all_modelling_tps = self.pred_tps
            dataset_args['time_steps'] = self.pred_tps
            if self.condition_encodings is not None:
                dataset_args['split_indices'] = self.train_indices
                dataset_args['conditions'] = (
                    self.conditions if self.condition_keys is not None else None
                )
                dataset_args['conditions_combined'] = (
                    self.conditions_combined
                    if self.condition_keys is not None
                    else None
                )
                self.train_dataset = CytoMeisterDataset(**dataset_args)
                if self.val_indices is not None:
                    dataset_args['split_indices'] = self.val_indices
                    self.val_dataset = CytoMeisterDataset(**dataset_args)
                else:
                    self.val_dataset = None
            else:
                dataset_args['split_indices'] = self.train_indices
                self.train_dataset = CytoMeisterDataset(**dataset_args)
                if self.val_indices is not None:
                    dataset_args['split_indices'] = self.val_indices
                    self.val_dataset = CytoMeisterDataset(**dataset_args)
                else:
                    self.val_dataset = None
        if stage == 'test' or stage is None:
            if self.context_tps is not None:
                # use only defined time steps for modelling to avoid data leakage
                self.all_modelling_tps = self.pred_tps + self.context_tps
            else:
                print('Define context_tps for testing')
                self.all_modelling_tps = self.total_tps

            dataset_args['time_steps'] = self.all_modelling_tps
            dataset_args['split_indices'] = self.test_indices
            if self.condition_encodings is not None:
                dataset_args['conditions'] = (
                    self.conditions if self.condition_keys is not None else None
                )
                dataset_args['conditions_combined'] = (
                    self.conditions_combined
                    if self.condition_keys is not None
                    else None
                )
                self.test_dataset = CytoMeisterDataset(**dataset_args)
            else:
                self.test_dataset = CytoMeisterDataset(**dataset_args)

    def train_dataloader(self):
        self.dataloader_kwargs['dataset'] = self.train_dataset
        self.dataloader_kwargs['collate_fn'] = self.collate
        data = DataLoader(**self.dataloader_kwargs)
        return data

    def val_dataloader(self):
        self.dataloader_kwargs['dataset'] = self.val_dataset
        self.dataloader_kwargs['shuffle'] = False
        self.dataloader_kwargs['collate_fn'] = self.collate
        if self.split:
            data = DataLoader(**self.dataloader_kwargs)
            return data
        else:
            return []

    def test_dataloader(self):
        self.dataloader_kwargs['dataset'] = self.test_dataset
        self.dataloader_kwargs['collate_fn'] = self.collate
        data = DataLoader(**self.dataloader_kwargs, drop_last=True)
        return data

    def collate(self, batch):
        src_dataset = [d['src_dataset'] for d in batch if 'src_dataset' in d]
        if src_dataset:
            src_input_batch_id = [torch.tensor(d['input_ids']) for d in src_dataset]
            src_length = torch.tensor([d['length'] for d in src_dataset])
            model_input_size = torch.max(src_length)
            src_input_batch_id = pad_tensor_list(
                src_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )
        else:
            src_input_batch_id, src_length = None, None
        src_counts = None
        if batch[0]['src_counts'] is not None:
            if isinstance(batch[0]['src_counts'], csr_matrix):
                src_counts = [torch.tensor(d['src_counts'].toarray()) for d in batch]

            else:
                src_counts = [torch.tensor(d['src_counts']) for d in batch]
            src_counts = torch.cat(src_counts, dim=0)
        if self.condition_encodings:
            condition = [d['conditions'] for d in batch]
            condition_combined = torch.stack([d['conditions_combined'] for d in batch])
        else:
            condition, condition_combined = None, None
        # compute tgt size factor
        out = {
            'src_input_ids': src_input_batch_id,
            'src_length': src_length,
            'src_counts': src_counts,
            'batch': condition,
            'combined_batch': condition_combined,
        }

        for time_step in self.all_modelling_tps:
            if batch[0]['tgt_counts_t1'] is not None:
                if isinstance(batch[0][f'tgt_counts_t{time_step}'], csr_matrix):
                    tgt_counts = [
                        torch.tensor(d[f'tgt_counts_t{time_step}'].toarray())
                        for d in batch
                    ]
                    tgt_size_factor = [
                        torch.tensor(
                            np.ravel(
                                d[f'tgt_counts_t{time_step}'].toarray().sum(axis=1)
                            )
                        )
                        for d in batch
                    ]
                else:
                    tgt_counts = [
                        torch.tensor(d[f'tgt_counts_t{time_step}']) for d in batch
                    ]
                    tgt_size_factor = [
                        torch.tensor(
                            np.ravel(d[f'tgt_counts_t{time_step}'].sum(axis=1))
                        )
                        for d in batch
                    ]
                out[f'tgt_counts_t{time_step}'] = torch.cat(tgt_counts, dim=0)
                out[f'tgt_size_factor_t{time_step}'] = torch.cat(tgt_size_factor, dim=0)
            # create input ids
            dataset = f'tgt_dataset_t{time_step}'
            out[f'tgt_input_ids_t{time_step}'] = [
                torch.tensor(d[dataset]['input_ids']) for d in batch
            ]
            length = [d[dataset]['length'] for d in batch]
            out[f'tgt_length_t{time_step}'] = torch.tensor(length)
            model_input_size = torch.max(out[f'tgt_length_t{time_step}'])
            out[f'tgt_cell_idx_t{time_step}'] = [
                d[dataset]['cell_pairing_index'] for d in batch
            ]
            if self.var_list is not None:
                for var in self.var_list:
                    out[f'{var}_t{time_step}'] = [d[dataset][var] for d in batch]

            out[f'tgt_input_ids_t{time_step}'] = pad_tensor_list(
                out[f'tgt_input_ids_t{time_step}'],
                self.max_len,
                self.pad_token_id,
                model_input_size,
            )

        return out
