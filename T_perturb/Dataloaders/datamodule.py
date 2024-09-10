import pickle
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

# import scanpy as sc
import torch
from datasets import DatasetDict
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pandas import Categorical
from pytorch_lightning import LightningDataModule

# from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset  # ConcatDataset,

# from warnings import warn


# from T_perturb.src.utils import weighted_sampler


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
        tgt_dataset: DatasetDict,
        pairing_metadata: Dict,
        split_indices: List,
    ):
        super().__init__()
        self.src_dataset = src_dataset
        self.pairing_metadata = pairing_metadata
        # subset tgt dataset based on split indices
        self.tgt_dataset = tgt_dataset.select(indices=split_indices)
        # # Get the indices from the datasets
        # src_indices = list(range(len(src_dataset)))
        # tgt_indices = list(range(len(tgt_dataset)))
        # src_len = len(self.src_dataset)
        self.tgt_len = len(self.tgt_dataset)

    def __getitem__(self, ind):
        tmp_dataset_id = self.tgt_dataset[ind]['dataset_id']
        # subset src dataset based on tgt dataset
        tmp_src_indices = self.pairing_metadata[tmp_dataset_id]
        # sample from tmp_src_id
        tmp_src_ind = np.random.choice(tmp_src_indices, 1)[0]
        out = {
            'src_dataset': self.src_dataset[int(tmp_src_ind)],
            'tgt_dataset': self.tgt_dataset[ind],
        }
        return out

    def __len__(self):
        return self.tgt_len


# two dataloader vs one dataloader
class CellGenDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_dataset: DatasetDict,
        pairing_metadata: Dict,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        split: bool = False,
        # condition_keys: Optional[list] = None,
        # condition_encodings: Optional[dict] = None,
        # conditions: Optional[torch.Tensor] = None,
        # conditions_combined: Optional[torch.Tensor] = None,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
        test_indices: Optional[list[int]] = None,
        # train_dict: Optional[Dict] = None,
        # val_dict: Optional[Dict] = None,
        # test_dict: Optional[Dict] = None,
        var_list: Optional[list] = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for CellGen tokenised data.
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.pairing_metadata = pairing_metadata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = max_len
        self.dataset = None
        # self.condition_keys = condition_keys
        # self.condition_encodings = condition_encodings
        # self.conditions = conditions
        # self.conditions_combined = conditions_combined
        # train test split
        self.split = split
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        # self.train_dict = train_dict
        # self.val_dict = val_dict
        # self.test_dict = test_dict
        self.var_list = var_list
        # create condition encoder for categorical variables in
        # form of dictionary with key: value pairs based on condition_keys
        self.train_dataset = None

    def setup(self, stage=None):
        dataset_params = {
            'src_dataset': self.src_dataset,
            'tgt_dataset': self.tgt_dataset,
            'pairing_metadata': self.pairing_metadata,
        }
        if stage == 'fit' or stage is None:
            self.train_dataset = CellGenDataset(
                split_indices=self.train_indices, **dataset_params
            )
            if self.val_indices is not None:
                self.val_dataset = CellGenDataset(
                    split_indices=self.val_indices, **dataset_params
                )
            else:
                self.val_dataset = None
        if stage == 'test' or stage is None:
            self.test_dataset = CellGenDataset(
                split_indices=self.test_indices, **dataset_params
            )

    def train_dataloader(self):
        # train_sampler = weighted_sampler(train_dataset)
        data = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            # sampler=train_sampler,
        )
        return data

    def val_dataloader(self):
        if self.split:
            if self.var_list is not None:
                data = DataLoader(
                    dataset=self.val_dataset,
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
        data = DataLoader(
            dataset=self.test_dataset,
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
        # if batch[0]['src_counts'] is not None:
        #     if isinstance(batch[0]['src_counts'], csr_matrix):
        #         src_counts = [torch.tensor(d['src_counts'].A) for d in batch]
        #     else:
        #         src_counts = [torch.tensor(d['src_counts']) for d in batch]
        #     src_counts = torch.cat(src_counts, dim=0)
        # else:
        #     src_counts = None

        # if self.condition_encodings:
        #     condition = [d['conditions'] for d in batch]
        #     condition_combined = torch.stack(
        #         [d['conditions_combined'] for d in batch]
        #         )
        # else:
        #     condition, condition_combined = None, None

        # tgt
        tgt_input_ids = [torch.tensor(d['tgt_dataset']['input_ids']) for d in batch]
        tgt_length = torch.tensor([d['tgt_dataset']['length'] for d in batch])
        tgt_input_ids = pad_tensor_list(
            tgt_input_ids, self.max_len, self.pad_token_id, model_input_size
        )
        # model_input_size = torch.max(tgt_length)
        # if batch[0]['tgt_counts'] is not None:
        #     if isinstance(batch[0]['tgt_counts'], csr_matrix):
        #         tgt_counts_matrices = [d['tgt_counts'].A for d in batch]
        #     else:
        #         tgt_counts_matrices = [d['tgt_counts'] for d in batch]

        #     tgt_counts = [torch.tensor(matrix) for matrix in tgt_counts_matrices]
        #     tgt_size_factor = [
        #         torch.tensor(np.ravel(matrix.sum(axis=1)))
        #         for matrix in tgt_counts_matrices
        #     ]
        #     tgt_counts = torch.cat(tgt_counts, dim=0)
        #     tgt_size_factor = torch.cat(tgt_size_factor, dim=0)
        # else:
        #     tgt_counts = None
        #     tgt_size_factor = None
        tgt_cell_idx = [d['tgt_dataset']['cell_pairing_index'] for d in batch]
        out = {
            'src_input_ids': src_input_ids,
            'src_length': src_length,
            # 'src_counts': src_counts,
            # 'batch': condition,
            # 'combined_batch': condition_combined,
            'tgt_input_ids': tgt_input_ids,
            'tgt_length': tgt_length,
            # 'tgt_counts': tgt_counts,
            # 'tgt_size_factor': tgt_size_factor,
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
            out['moe_one_hot_categories'] = one_hot_labels
            out['moe_categories'] = Categorical(categories).categories
        return out


if __name__ == '__main__':
    from datasets import load_from_disk

    # test dataloader
    # data_module = CellGenDataModule(
    #     src_dataset=(
    #         '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    #         'T_perturb/T_perturb/pp/res/dataset/'
    #         'cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset'
    #     ),
    #     tgt_datasets=(
    #         '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    #         'T_perturb/T_perturb/pp/res/dataset/'
    #         'cytoimmgen_tokenised_degs_stratified_pairing_16h.dataset'
    #     ),
    #     max_len=334,
    # )
    # data_module.setup()
    # dataloader = data_module.train_dataloader()
    # # iterate through batches
    # train_iterator = iter(dataloader)
    # batch = next(train_iterator)
    # print(batch['tgt_input_ids'][:20, :20])
    # print(len(batch['tgt_counts'][0]))
    src_dataset = load_from_disk(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/CellGen-reproducibility/covid_ipf_copd/'
        'processed_data/dataset_hvg_src/control.dataset'
    )
    tgt_dataset = load_from_disk(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/CellGen-reproducibility/covid_ipf_copd/'
        'processed_data/dataset_hvg_tgt/disease.dataset'
    )
    with open(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/CellGen-reproducibility/'
        'covid_ipf_copd/processed_data/metadata.pkl',
        'rb',
    ) as f:
        ipf_cell_pairing = pickle.load(f)

    dataset = CellGenDataset(
        src_dataset,
        tgt_dataset,
        ipf_cell_pairing,
        split_indices=list(range(10)),
    )
    dataset[0]
