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

    # def pad_tensor(self,tensor, max_len=2048):
    #     tensor = torch.nn.functional.pad(tensor, 
    #                                      pad=(0,max_len - tensor.numel()),
    #                                      mode='constant',
    #                                      value=self.pad_token_id)
    #     return tensor

#two dataloader vs one dataloader
class GeneformerDataModule(LightningDataModule):
    def __init__(self,
                 folder= "./data/tokenized.dataset",
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
        self.folder = folder
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
        self.dataset = GeneformerDataset(self.folder, shuffle=self.shuffle)

    def train_dataloader(self):
        return DataLoader(self.dataset, collate_fn= self.src_collate, batch_size=self.batch_size
                          , shuffle=self.shuffle, num_workers=self.num_workers)

    def src_collate(self,batch):
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
        return {"src_input_id": input_batch_id.clone().detach(), 
                "src_length" : length.clone().detach(), 
                "src_cell_type" : cell_type,
                "src_time_point" : time_point,
                "src_Donor" : Donor,
                "src_attention_mask" : self.gen_attention_mask(length), #no attention mask needed for tgt
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
    data_module=GeneformerDataModule("./res/dataset/cytoimmgen_tokenised_degs.dataset", max_len=334) 
    data_module.setup()
    dataloader = data_module.train_dataloader()
    #iterate through batches
    train_iterator = iter(dataloader)
    batch = next(train_iterator)
    print(batch)