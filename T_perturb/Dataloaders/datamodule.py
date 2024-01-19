import pickle

import torch
from datasets import load_from_disk
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class GeneformerDataset(Dataset):
    def __init__(
        self,
        src_folder='./data/tokenized.dataset',
        tgt_folder='./data/tokenized.dataset',
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
        self.src_data = load_from_disk(src_folder)
        self.tgt_data = load_from_disk(tgt_folder)
        # with open(token_dictionary_file, "rb") as f:
        #     self.gene_token_dict = pickle.load(f)
        # self.pad_token_id = self.gene_token_dict.get("<pad>")

    def __len__(self):
        if len(self.src_data) != len(self.tgt_data):
            Warning('src and tgt dataset have different length')
        return min(len(self.src_data), len(self.tgt_data))

    def __getitem__(self, ind):
        return {'src': self.src_data[ind], 'tgt': self.tgt_data[ind]}


# two dataloader vs one dataloader
class GeneformerDataModule(LightningDataModule):
    def __init__(
        self,
        src_folder='./data/tokenized.dataset',
        tgt_folder='./data/tokenized.dataset',
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
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
        self.pad_token_id = self.gene_token_dict.get('<pad>')
        self.max_len = max_len
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = GeneformerDataset(
            src_folder=self.src_folder,
            tgt_folder=self.tgt_folder,
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
            self.src_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return data

    def collate(self, batch):
        if any('src' in item for item in batch):
            src_input_batch_id = [torch.tensor(d['src']['input_ids']) for d in batch]
            src_length = torch.stack([torch.tensor(d['src']['length']) for d in batch])
            model_input_size = torch.max(src_length)
            src_cell_type = [d['src']['Cell_type'] for d in batch]
            src_time_point = ([d['src']['Time_point'] for d in batch],)
            src_donor = ([d['src']['Donor'] for d in batch],)
            src_input_batch_id = pad_tensor_list(
                src_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )
        if any('tgt' in item for item in batch):
            tgt_input_batch_id = [torch.tensor(d['tgt']['input_ids']) for d in batch]
            tgt_length = torch.stack([torch.tensor(d['tgt']['length']) for d in batch])
            model_input_size = torch.max(tgt_length)
            tgt_cell_type = [d['tgt']['Cell_type'] for d in batch]
            tgt_time_point = ([d['tgt']['Time_point'] for d in batch],)
            tgt_donor = ([d['tgt']['Donor'] for d in batch],)
            tgt_input_batch_id = pad_tensor_list(
                tgt_input_batch_id, self.max_len, self.pad_token_id, model_input_size
            )
        return {
            'src_input_ids': src_input_batch_id,
            'src_length': src_length,
            'src_cell_type': src_cell_type,
            'src_time_point': src_time_point,
            'src_donor': src_donor,
            'tgt_input_ids': tgt_input_batch_id,
            'tgt_length': tgt_length,
            'tgt_cell_type': tgt_cell_type,
            'tgt_time_point': tgt_time_point,
            'tgt_donor': tgt_donor,
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
        src_folder='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_0h.dataset',
        tgt_folder='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_16h.dataset',
        max_len=334,
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()
    # iterate through batches
    train_iterator = iter(dataloader)
    batch = next(train_iterator)
    print(batch)
