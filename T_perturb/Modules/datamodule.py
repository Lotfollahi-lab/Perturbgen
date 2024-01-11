import os
from pathlib import Path
import scanpy as sc
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from geneformer import TranscriptomeTokenizer
from datasets import load_from_disk
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE


class GeneformerDataset(Dataset):
    def __init__(self,
                 folder = "./data/tokenized.dataset",
                 shuffle=False,
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
        self.dataset = load_from_disk(folder)
        # with open(token_dictionary_file, "rb") as f:
        #     self.gene_token_dict = pickle.load(f)
        # self.pad_token_id = self.gene_token_dict.get("<pad>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        # Success
        return self.dataset[ind]

#two dataloader vs one dataloader
class GeneformerDataModule(LightningDataModule):
    def __init__(self,
                 src_folder= "./data/tokenized.dataset",
                 tgt_folder= "./data/tokenized.dataset",
                 batch_size=3,
                 num_workers=0,
                 shuffle=False,
                 max_len=2048,
                 ):
        """
        Description:
        ------------
        Custom datamodule for Geneformer tokenised data.
        """
        super().__init__()
        self.src_folder = src_folder
        self.tgt_folder = tgt_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)
        self.pad_token_id = self.gene_token_dict.get("<pad>")
        self.max_len = max_len
        self.dataset = None

    def setup(self, stage=None):
        self.src_dataset = GeneformerDataset(
            folder=self.src_folder,
            shuffle=self.shuffle,
            )
        self.tgt_dataset = GeneformerDataset(
            folder=self.tgt_folder,
            shuffle=self.shuffle,
            )

    def train_dataloader(self):
        return {
            "src": DataLoader(
                self.src_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                collate_fn=self.src_collate,
                ),
            "tgt": DataLoader(
                self.tgt_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                collate_fn=self.tgt_collate,
                )
        }
    def src_collate(self, batch):
        batch = batch
        model_input_size = self.max_len
        input_batch_id = [torch.tensor(d["input_ids"]) for d in batch]
        length = torch.stack([torch.tensor(d["length"]) for d in batch])
        cell_type = [d["Cell_type"] for d in batch]
        time_point = [d["Time_point"] for d in batch],
        Donor = [d["Donor"] for d in batch],
        input_batch_id = pad_tensor_list(
            input_batch_id,
            self.max_len,
            self.pad_token_id,
            model_input_size)
        return {"input_id": input_batch_id.clone().detach(), 
                "length" : length.clone().detach(), 
                "cell_type" : cell_type,
                "time_point" : time_point,
                "Donor" : Donor,
                "attention_mask" : self.gen_attention_mask(length), #no attention mask needed for tgt
                }
    def tgt_collate(self, batch):
        batch = batch
        model_input_size = self.max_len
        input_batch_id = [torch.tensor(d["input_ids"]) for d in batch]
        length = torch.stack([torch.tensor(d["length"]) for d in batch])
        cell_type = [d["Cell_type"] for d in batch]
        time_point = [d["Time_point"] for d in batch],
        Donor = [d["Donor"] for d in batch],
        input_batch_id = pad_tensor_list(
            input_batch_id,
            self.max_len,
            self.pad_token_id,
            model_input_size)
        return {"input_id": input_batch_id.clone().detach(), 
                "length" : length.clone().detach(), 
                "cell_type" : cell_type,
                "time_point" : time_point,
                "Donor" : Donor
                }

    def gen_attention_mask(self, length):
        attention_mask = [[1] * original_len
                          + [0] * (self.max_len - original_len)
                          if original_len <= self.max_len
                          else [1] * self.max_len
                          for original_len in length]

        return torch.tensor(attention_mask)
    
if __name__ ==  "__main__":
    #test dataloader
    data_module=GeneformerDataModule(
        src_folder= "/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_0h.dataset",
        tgt_folder= "/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_16h.dataset",
        max_len=334
        )
    data_module.setup()
    dataloader = data_module.train_dataloader()
    #iterate through batches
    train_iterator = iter(dataloader["src"])
    batch = next(train_iterator)
    print(batch)