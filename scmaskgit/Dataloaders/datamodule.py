import pickle
from typing import Optional
from warnings import warn

import numpy as np
import torch
from datasets import DatasetDict
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

def pad_tensor(tensor, pad_token_id, max_len):
    tensor = torch.nn.functional.pad(
        tensor, pad=(0, max_len - tensor.numel()), mode="constant", value=pad_token_id
    )

    return tensor
def pad_tensor_list(
    tensor_list,
    dynamic_or_constant,
    pad_token_id,
    model_input_size,
    dim=None,
    padding_func=None,
):
    # determine maximum tensor length
    
    max_len = model_input_size
   

    # pad all tensors to maximum length
    if dim is None:
        tensor_list = [
            pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list
        ]
    else:
        tensor_list = [
            padding_func(tensor, pad_token_id, max_len, dim) for tensor in tensor_list
        ]
    # return stacked tensors
    return torch.stack(tensor_list)
    


class CellGenDataset(Dataset):
    def __init__(
        self,
        src_dataset: DatasetDict,
        time_steps: list = [1, 2],
        src_counts: Optional[np.ndarray] = None,
        tgt_counts_dict: Optional[np.ndarray] = None,
        split_indices: Optional[list] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        condition_encodings: Optional[dict] = None,
    ):
        super().__init__()
        self.src_dataset = src_dataset
        self.src_counts = src_counts

        self.conditions = conditions
        self.conditions_combined = conditions_combined
        if split_indices is not None:
            self.src_dataset = src_dataset.select(split_indices)
            if self.src_counts is not None:
                self.src_counts = self.src_counts[split_indices, :]
        src_len = len(self.src_dataset)
       
        self.dataset_length = src_len

    def __getitem__(self, ind):
        out = {
            'src_dataset': self.src_dataset[ind],
            'src_counts': self.src_counts[ind] if self.src_counts is not None else None,
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }
        return out

    def __len__(self):
        return len(self.src_dataset)


class CellGenDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = True,
        max_len: int = 2048,
        split: bool = False,
        pred_tps: list = [1, 2],
        n_total_tps: int = 4,
        src_counts: Optional[np.ndarray] = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions: Optional[torch.Tensor] = None,
        conditions_combined: Optional[torch.Tensor] = None,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
        test_indices: Optional[list[int]] = None,
        var_list: Optional[list] = None,
        context_tps: Optional[list] = None,
        test_dataset: Optional[DatasetDict] = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for CellGen tokenised data.
        """
        super().__init__()
        print('Start datamodule')
        self.src_dataset = src_dataset
        self.test_dataset = test_dataset
        self.src_counts = src_counts
        self.dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
        }
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
        self.pred_tps = pred_tps
        self.context_tps = context_tps
        self.total_tps = list(range(1, n_total_tps + 1))
        self.var_list = var_list
        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys

    def setup(self, stage=None):
        if self.context_tps is not None:
            all_modelling_tps = self.pred_tps + self.context_tps
            self.all_modelling_tps = list(set(all_modelling_tps))
        else:
            self.all_modelling_tps = self.pred_tps
        dataset_args = {
            'src_dataset': self.src_dataset,
            'src_counts': self.src_counts,
        }
        # Assign train/val datasets for use in dataloaders
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = CellGenDataset(**dataset_args)
            if self.val_indices is not None:
                self.val_dataset = CellGenDataset(**dataset_args)
            else:
                self.val_dataset = None
        if stage == 'test' or stage is None:
            # use all time steps to provide as context
            test_dataset_args = {
            'src_dataset': self.test_dataset,
            }
            self.test_dataset = CellGenDataset(**test_dataset_args)

    def train_dataloader(self):
        data = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            drop_last = True,
        )
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
        self.dataloader_kwargs['shuffle'] = False
        self.dataloader_kwargs['collate_fn'] = self.collate
        data = DataLoader(**self.dataloader_kwargs)
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
        
        # compute tgt size factor        
        out = {
            'src_input_ids': src_input_batch_id,
            'src_length': src_length,
        }
        if self.var_list is not None:
                for var in self.var_list:
                    out[f'{var}'] = [d[var] for d in src_dataset]
        return out
