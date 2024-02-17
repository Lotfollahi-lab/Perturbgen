import pickle
from typing import Optional

import anndata as ad
import numpy as np

# import scanpy as sc
import torch
from datasets import load_from_disk
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

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


class GeneformerDataset(Dataset):
    def __init__(
        self,
        src_dataset_folder: str = './data/tokenized.dataset',
        tgt_dataset_folder: str = './data/tokenized.dataset',
        shuffle: bool = False,
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
        self.src_data = load_from_disk(src_dataset_folder)
        self.tgt_data = load_from_disk(tgt_dataset_folder)
        self.tgt_adata = tgt_adata
        self.size_factor = np.ravel(tgt_adata.X.sum(axis=1))
        self.conditions = conditions
        print(self.conditions)
        self.conditions_combined = conditions_combined
        print(self.conditions_combined)
        self.condition_encodings = condition_encodings

        # with open(token_dictionary_file, "rb") as f:
        #     self.gene_token_dict = pickle.load(f)
        # self.pad_token_id = self.gene_token_dict.get("<pad>")

    def __len__(self):
        if len(self.src_data) != len(self.tgt_data):
            Warning('src and tgt dataset have different length')
        return min(len(self.src_data), len(self.tgt_data))

    def __getitem__(self, ind):
        return {
            'src_dataset': self.src_data[ind],
            'tgt_dataset': self.tgt_data[ind],
            'tgt_adata': self.tgt_adata[ind],
            'tgt_size_factor': self.size_factor[ind],
            'conditions': self.conditions[ind] if self.conditions is not None else None,
            'conditions_combined': self.conditions_combined[ind]
            if self.conditions_combined is not None
            else None,
        }


# two dataloader vs one dataloader
class GeneformerDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset_folder: str = './data/tokenized.dataset',
        tgt_dataset_folder: str = './data/tokenized.dataset',
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        loss_mode: str = 'mse',
        tgt_adata: ad.AnnData = None,
        condition_keys: Optional[list] = None,
        condition_encodings: Optional[dict] = None,
        conditions_combined_encodings: Optional[dict] = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for Geneformer tokenised data.
        """
        super().__init__()
        self.src_dataset_folder = src_dataset_folder
        self.tgt_dataset_folder = tgt_dataset_folder
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
            self.dataset = GeneformerDataset(
                src_dataset_folder=self.src_dataset_folder,
                tgt_dataset_folder=self.tgt_dataset_folder,
                tgt_adata=self.tgt_adata,
                shuffle=self.shuffle,
                conditions=self.conditions if self.condition_keys is not None else None,
                conditions_combined=self.conditions_combined
                if self.condition_keys is not None
                else None,
            )
        else:
            self.dataset = GeneformerDataset(
                src_dataset_folder=self.src_dataset_folder,
                tgt_dataset_folder=self.tgt_dataset_folder,
                tgt_adata=self.tgt_adata,
                shuffle=self.shuffle,
            )
        # if stage == 'fit' or stage is None:
        #     self.dataset = self.src_dataset + self.tgt_dataset
        # if stage == 'test' or stage is None:
        #     self.dataset = self.tgt_dataset

    def train_test_split(self):
        pass

    def train_dataloader(self):
        data = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return data

    def test_dataloader(self):
        data = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
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

        if any('tgt_adata' in item for item in batch):
            tgt_counts = [d['tgt_adata'].X for d in batch]
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


if __name__ == '__main__':
    # test dataloader
    data_module = GeneformerDataModule(
        src_dataset_folder=(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            'T_perturb/T_perturb/pp/res/dataset/'
            'cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset'
        ),
        tgt_dataset_folder=(
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
