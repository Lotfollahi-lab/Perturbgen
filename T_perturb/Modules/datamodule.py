import os
from pathlib import Path
import scanpy as sc
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from geneformer import TranscriptomeTokenizer
from datasets import Dataset, load_from_disk
from geneformer.in_silico_perturber import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE


class GeneformerDataset(Dataset):
    def __init__(self,
                 folder = "./data/tokenized.dataset",
                 shuffle=False,
                 var_to_keep: list = None,
                 ):
        """
        Description:
        ------------
        This class load adata object from disk and tokenize it using geneformer tokenizer.

        """
        if var_to_keep: 
            self.var_to_keep = {v : v for v in var_to_keep}
        else:
            raise ValueError("var_to_keep must be provided")

        self.tokenizer = TranscriptomeTokenizer(self.var_to_keep, nproc=16)
        self.shuffle = shuffle
        self.adata = load_from_disk(folder)
        # with open(token_dictionary_file, "rb") as f:
        #     self.gene_token_dict = pickle.load(f)
        # self.pad_token_id = self.gene_token_dict.get("<pad>")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, ind):
        # Success
        return self.adata[ind]

    def tokenize(self, adata, file = 'data/preprocess.h5ad'):
        adata.write_h5ad(file)
        self.tokenizer.tokenize_data("./data",
                                "./data",
                                "tokenized",
                                file_format="h5ad")

    def pad_tensor(self,tensor, max_len=2048):
        max_len = get_model_input_size('GeneClassifier')
        tensor = torch.nn.functional.pad(tensor, 
                                         pad=(0,max_len - tensor.numel()),
                                         mode='constant',
                                         value=self.pad_token_id)
        return tensor

class GeneformerDataModule(LightningDataModule):
    def __init__(self,
                 folder= "./data/tokenized.dataset",
                 batch_size=3,
                 num_workers=0,
                 shuffle=False,
                 tokenizer=None
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
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
        self.max_len = 2048

    def prepare_data(self):
        assert self.h5ad_path.exists(), ".h5ad file does not exist"

    def setup(self, stage=None):
        self.dataset = GeneformerDataset(self.folder, shuffle=self.shuffle)

    def train_dataloader(self):
        return DataLoader(self.dataset, collate_fn= self.custom_collate, batch_size=self.batch_size
                          , shuffle=self.shuffle, num_workers=self.num_workers)

    def custom_collate(self,batch):
        model_input_size = 2048
        input_batch_id = [torch.tensor(d["input_ids"]) for d in batch]
        length = torch.stack([torch.tensor(d["length"]) for d in batch])
        cell = [d["cell_type"] for d in batch]
        input_batch_id = pad_tensor_list(
            input_batch_id,
            2048,
            self.pad_token_id,
            model_input_size)
        return {"input_id": input_batch_id.clone().detach()
                , "length" : length.clone().detach()
                , "cell_type" : cell
                , "attention_mask" : self.gen_attention_mask(length)
                }

    def gen_attention_mask(self, length):
        attention_mask = [[1] * original_len
                          + [0] * (self.max_len - original_len)
                          if original_len <= self.max_len
                          else [1] * self.max_len
                          for original_len in length]

        return torch.tensor(attention_mask)