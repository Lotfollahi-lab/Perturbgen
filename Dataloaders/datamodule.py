import pickle
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
import mygene


# import scanpy as sc
import torch
import anndata as ad
from datasets import DatasetDict
from geneformer.perturber_utils import pad_tensor_list
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pandas import Categorical
from pytorch_lightning import LightningDataModule
from scipy.sparse import csr_matrix

from T_perturb.gene_embedding import extract_gene_embeddings
from T_perturb.Modules.T_model_GFE import Geneformerwrapper

import scipy
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


    
# ### GENEFORMER EMBEDDINGS ###
# class CellGenDataset(Dataset):
#     def __init__(
#         self,
#         src_dataset: DatasetDict,
#         tgt_dataset: DatasetDict,
#         src_counts: np.ndarray,
#         tgt_counts_dict: Dict[str, np.ndarray],
#         tgt_adata: ad.AnnData,
#         split_indices: List,
#     ):
#         super().__init__()
#         self.src_dataset = src_dataset
#         # self.pairing_metadata = pairing_metadata
#         # subset tgt dataset based on split indices
#         self.tgt_dataset = tgt_dataset.select(indices=split_indices)
#         # # Get the indices from the datasets
#         # src_indices = list(range(len(src_dataset)))
#         # tgt_indices = list(range(len(tgt_dataset)))
#         # src_len = len(self.src_dataset)
#         self.tgt_len = len(self.tgt_dataset)
#         self.tgt_adata = tgt_adata

#     def __getitem__(self, ind):
#         # tmp_dataset_id = self.tgt_dataset[ind]['dataset_id']
#         # subset src dataset based on tgt dataset
#         # tmp_src_indices = self.pairing_metadata[tmp_dataset_id]
#         # sample from tmp_src_id
#         tmp_src_ind = np.random.randint(len(self.src_dataset))
        

#         # Extract perturbation metadata
#         perturbation_info = self.tgt_dataset[ind].get('perturbation', [])
#         nperts_info = self.tgt_dataset[ind].get('nperts', 0)
        
        
#         cell_pairing_index = self.tgt_dataset[ind]['cell_pairing_index']
        
#         # Use cell_pairing_index to get counts from adata.X
#         true_counts = self.tgt_adata.X[cell_pairing_index]

#         # Convert counts to tensor
#         true_counts = torch.tensor(true_counts.toarray()).squeeze(0)  # Assuming sparse matrix

        
#         # Process perturbation_info to ensure it's a list of gene names
#         if perturbation_info:
#             if isinstance(perturbation_info, str):
#                 perturbation_info = perturbation_info.split('+')  # Split by '+'
#             elif isinstance(perturbation_info, list):
#                 # Ensure splitting for each element
#                 perturbation_info = [gene for genes in perturbation_info for gene in genes.split('+')]
#         else:
#             perturbation_info = []
            
#         # Initialize gene_embeddings tensor of shape (2, embedding_dim)
#         embedding_dim = 512  # Assuming Geneformer embedding dimension is 512
#         max_nperts = 2
#         gene_embeddings = torch.zeros((max_nperts, embedding_dim))

                
#         ## ADDING PERTURBATION EMBEDDINGS
        
#         if nperts_info > 0:
            
#             ## NORMAL LOADING GENEFORMER ###
#             # Load Geneformer wrapper in frozen mode
#             geneformer_frozen = Geneformerwrapper(
#                 output_attentions=False,
#                 output_hidden_states=True,  # Ensure hidden states are returned
#                 mode='GF_frozen'  # Mode to use frozen embeddings
#             )

#             # Load the token-to-gene dictionary
#             with open('/lustre/scratch126/cellgen/team205/bair/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl', 'rb') as file:
#                 token_to_gene_dict = pickle.load(file)

#             # Extract gene embeddings for the perturbed genes
#             embeddings = extract_gene_embeddings(perturbation_info, token_to_gene_dict, geneformer_frozen)
#             # embeddings shape: (1, nperts, embedding_dim)

#             embeddings = embeddings.squeeze(0)  # Shape: (nperts, embedding_dim)

#             # Pad embeddings to shape (max_nperts, embedding_dim)
#             nperts = embeddings.shape[0]
#             if nperts < max_nperts:
#                 # Pad with zeros
#                 padding = torch.zeros((max_nperts - nperts, embedding_dim))
                
#                 # print(f"embeddings shape: {embeddings.shape}")
#                 # print(f"padding shape: {padding.shape}")
                
#                 embeddings = torch.cat([embeddings, padding], dim=0)
#             elif nperts > max_nperts:
#                 # If more than max_nperts perturbations, truncate
#                 embeddings = embeddings[:max_nperts, :]

#             gene_embeddings = embeddings  # Shape: (max_nperts, embedding_dim)
#         else:
#             pass # No perturbations
        
#         # print(f'self.tgt_dataset: {self.tgt_dataset[ind]}')
        
#         out = {
#             'src_dataset': self.src_dataset[int(tmp_src_ind)],
#             'tgt_dataset': self.tgt_dataset[ind],
#             'perturbation': perturbation_info,  # List of perturbed genes
#             'nperts': nperts_info,  # Number of perturbations
#             'perturbed_embeddings': gene_embeddings,  # Gene embeddings of shape (max_nperts, embedding_dim)
#             'tgt_counts': true_counts,  # Include true counts
#         }
#         return out

#     def __len__(self):
#         return self.tgt_len



def get_symbol_to_ensembl_mapping(gene_symbols):
    mg = mygene.MyGeneInfo()
    # Query MyGene.info to get Ensembl IDs
    query_results = mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene', species='human')
    symbol_to_ensembl = {}
    for entry in query_results:
        symbol = entry['query']
        ensembl_ids = []
        if 'ensembl' in entry:
            ensembl = entry['ensembl']
            if isinstance(ensembl, list):
                ensembl_ids = [item['gene'] for item in ensembl if 'gene' in item]
            elif isinstance(ensembl, dict) and 'gene' in ensembl:
                ensembl_ids = [ensembl['gene']]
        symbol_to_ensembl[symbol] = ensembl_ids
    return symbol_to_ensembl
    

### ONE HOT EMBEDDINGS ###
class CellGenDataset(Dataset):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_dataset: DatasetDict,
        src_counts: np.ndarray,
        tgt_counts_dict: Dict[str, np.ndarray],
        tgt_adata: ad.AnnData,
        split_indices: List,
        perturbation_to_index: Dict[str, int],
        num_perturbations: int,
    ):
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset.select(indices=split_indices)
        self.tgt_len = len(self.tgt_dataset)
        self.tgt_adata = tgt_adata
        self.perturbation_to_index = perturbation_to_index
        self.num_perturbations = num_perturbations



    def __getitem__(self, ind):
        tmp_src_ind = np.random.randint(len(self.src_dataset))
        
        # Extract perturbation information
        perturbation_info = self.tgt_dataset[ind].get('perturbation', [])
        nperts_info = self.tgt_dataset[ind].get('nperts', 0)

        # Process perturbation_info into a list
        if perturbation_info:
            if isinstance(perturbation_info, str):
                perturbation_info = perturbation_info.split('+')
            elif isinstance(perturbation_info, list):
                perturbation_info = [
                    gene for genes in perturbation_info for gene in genes.split('+')
                ]
        else:
            perturbation_info = []

        # Initialize one-hot perturbation vector
        perturbation_vector = torch.zeros(self.num_perturbations)
        if nperts_info > 0:
            for perturbation in perturbation_info:
                idx = self.perturbation_to_index.get(perturbation)
                if idx is not None:
                    perturbation_vector[idx] = 1

        # Prepare the output dictionary
        out = {
            'src_dataset': self.src_dataset[int(tmp_src_ind)],
            'tgt_dataset': self.tgt_dataset[ind],
            'perturbation_vector': perturbation_vector,  # One-hot encoded vector
            'nperts': nperts_info,  # Number of perturbations
        }
        return out

    def __len__(self):
        return self.tgt_len
    
    
# ## RANDOM EMBEDDINGS #####
# class CellGenDataset(Dataset):
#     def __init__(
#         self,
#         src_dataset: DatasetDict,
#         tgt_dataset: DatasetDict,
#         src_counts: np.ndarray,
#         tgt_counts_dict: Dict[str, np.ndarray],
#         tgt_adata: ad.AnnData,
#         split_indices: List,
#     ):
#         super().__init__()
#         self.src_dataset = src_dataset
#         self.tgt_dataset = tgt_dataset.select(indices=split_indices)
#         self.tgt_len = len(self.tgt_dataset)
#         self.src_counts = src_counts
#         self.tgt_counts_dict = tgt_counts_dict
#         self.tgt_adata = tgt_adata  # Store the anndata object
        
#         print(f'tgt_counts_dict keys: {tgt_counts_dict.keys()}')


#     def __getitem__(self, ind):
#         tmp_src_ind = np.random.randint(len(self.src_dataset))

#         perturbation_info = self.tgt_dataset[ind].get('perturbation', [])
#         nperts_info = self.tgt_dataset[ind].get('nperts', 0)

#         # Process perturbation embeddings as before
#         embedding_dim = 512
#         max_nperts = 2
#         gene_embeddings = torch.zeros((max_nperts, embedding_dim))

#         if nperts_info > 0:
#             embeddings = torch.randn(1, nperts_info, embedding_dim)
#             embeddings = embeddings.squeeze(0)

#             nperts = embeddings.shape[0]
#             if nperts < max_nperts:
#                 padding = torch.zeros((max_nperts - nperts, embedding_dim))
#                 embeddings = torch.cat([embeddings, padding], dim=0)
#             elif nperts > max_nperts:
#                 embeddings = embeddings[:max_nperts, :]

#             gene_embeddings = embeddings
#         else:
#             pass  # No perturbations

#         # Retrieve counts for the target cell using cell_pairing_index
#         cell_pairing_index = self.tgt_dataset[ind]['cell_pairing_index']
        
#         true_counts = self.tgt_adata.X[cell_pairing_index]

#         # Convert counts to torch tensor
#         if scipy.sparse.issparse(true_counts):
#             true_counts = torch.tensor(true_counts.toarray()).squeeze(0)
#         else:
#             true_counts = torch.tensor(true_counts).squeeze(0)

#         out = {
#             'src_dataset': self.src_dataset[int(tmp_src_ind)],
#             'tgt_dataset': self.tgt_dataset[ind],
#             'perturbation': perturbation_info,
#             'nperts': nperts_info,
#             'perturbed_embeddings': gene_embeddings,
#             'tgt_counts': true_counts,  # Include true counts
#         }
        
        
#         out['tgt_dataset'] = self.tgt_dataset[ind]
        
#         if (self.tgt_counts_dict is not None) and (
#                 'tgt_h5ad_t1' in self.tgt_counts_dict
#             ):
#                 out['tgt_counts'] = self.tgt_counts_dict['tgt_h5ad_t1'][ind]
#         return out

#     def __len__(self):
#         return self.tgt_len



# two dataloader vs one dataloader
class CellGenDataModule(LightningDataModule):
    def __init__(
        self,
        src_dataset: DatasetDict,
        tgt_adata: ad.AnnData,
        tgt_dataset: DatasetDict,
        src_counts: np.ndarray,
        tgt_counts_dict: Dict[str, np.ndarray],
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle: bool = False,
        max_len: int = 2048,
        split: bool = False,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
        test_indices: Optional[list[int]] = None,
        var_list: Optional[list] = None,
        # tokenid_to_genename_dict: Optional[Dict[int, str]] = None,  # Add this parameter


        # self,
        # src_dataset: DatasetDict,
        # tgt_dataset: DatasetDict,
        # tgt_adata: ad.AnnData,
        # # pairing_metadata: Dict,
        # batch_size: int = 64,
        # num_workers: int = 8,
        # shuffle: bool = False,
        # max_len: int = 2048,
        # split: bool = False,
        # # condition_keys: Optional[list] = None,
        # # condition_encodings: Optional[dict] = None,
        # # conditions: Optional[torch.Tensor] = None,
        # # conditions_combined: Optional[torch.Tensor] = None,
        # train_indices: Optional[list[int]] = None,
        # val_indices: Optional[list[int]] = None,
        # test_indices: Optional[list[int]] = None,
        # # train_dict: Optional[Dict] = None,
        # # val_dict: Optional[Dict] = None,
        # # test_dict: Optional[Dict] = None,
        # var_list: Optional[list] = None,
    ):
        """
        Description:
        ------------
        Custom datamodule for CellGen tokenised data.
        """
        super().__init__()
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        
        self.src_counts = src_counts
        self.tgt_counts_dict = tgt_counts_dict

        # self.pairing_metadata = pairing_metadata
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
        self.tgt_adata = tgt_adata
        self.test_dataset = None
        # self.tokenid_to_genename_dict = tokenid_to_genename_dict
        # self.symbol_to_ensembl = get_symbol_to_ensembl_mapping(list(self.tokenid_to_genename_dict.values()))


        self.perturbation_to_index = None
        self.num_perturbations = 0


    def setup(self, stage=None):



        perturbations = set()
        for sample in self.tgt_dataset:
            perturbation_info = sample.get('perturbation', [])
            if perturbation_info:
                if isinstance(perturbation_info, str):
                    perturbation_list = perturbation_info.split('+')
                elif isinstance(perturbation_info, list):
                    perturbation_list = [
                        gene for genes in perturbation_info for gene in genes.split('+')
                    ]
                perturbations.update(perturbation_list)

        self.perturbation_list = sorted(list(perturbations))
        self.perturbation_to_index = {
            p: i for i, p in enumerate(self.perturbation_list)
        }
        self.num_perturbations = len(self.perturbation_list)



        dataset_params = {
            'src_dataset': self.src_dataset,
            'tgt_dataset': self.tgt_dataset,
            'tgt_adata': self.tgt_adata,
            'src_counts': self.src_counts,
            'tgt_counts_dict': self.tgt_counts_dict,
            'perturbation_to_index': self.perturbation_to_index,
            'num_perturbations': self.num_perturbations,
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
    
    # CHATGPT
    def collate(self, batch):
        # src
        src_input_ids = [torch.tensor(d['src_dataset']['input_ids']) for d in batch]
        src_length = torch.tensor([d['src_dataset']['length'] for d in batch])
        model_input_size = torch.max(src_length)
        src_input_ids = pad_tensor_list(
            src_input_ids, self.max_len, self.pad_token_id, model_input_size
        )

        conditions_combined = [1 for d in batch]

        # tgt
        tgt_input_ids = [torch.tensor(d['tgt_dataset']['input_ids']) for d in batch]
        tgt_length = torch.tensor([d['tgt_dataset']['length'] for d in batch])
        tgt_input_ids = pad_tensor_list(
            tgt_input_ids, self.max_len, self.pad_token_id, model_input_size
        )
        tgt_cell_idx = [d['tgt_dataset']['cell_pairing_index'] for d in batch]

        # Gather perturbation embeddings
        perturbed_embeddings_list = [d['perturbed_embeddings'] for d in batch]
        perturbed_embeddings = torch.stack(perturbed_embeddings_list, dim=0)

        # Collect tgt_counts and compute size factors
        if batch[0]['tgt_counts'] is not None:
            if isinstance(batch[0]['tgt_counts'], csr_matrix):
                # Handle sparse matrix case
                tgt_counts_list = [torch.tensor(d['tgt_counts'].A).squeeze() for d in batch]
                tgt_size_factor_list = [
                    torch.tensor(d['tgt_counts'].A.sum()).unsqueeze(0) for d in batch
                ]
            else:
                # Handle dense tensor case
                tgt_counts_list = [d['tgt_counts'].squeeze() for d in batch]
                tgt_size_factor_list = [
                    torch.tensor(d['tgt_counts'].sum()).unsqueeze(0) for d in batch
                ]

            # Stack counts and size factors into tensors
            tgt_counts = torch.stack(tgt_counts_list, dim=0).float()  # Shape: [batch_size, num_genes]
            tgt_size_factor = torch.stack(tgt_size_factor_list, dim=0).float().squeeze(1)  # Shape: [batch_size]

            # Verify shapes
            print("tgt_counts shape:", tgt_counts.shape)
            assert tgt_counts.shape[1] == self.num_genes, "Mismatch in gene count dimensions"
        else:
            tgt_counts = None
            tgt_size_factor = None

        perturbation_vectors = torch.stack([d['perturbation_vector'] for d in batch], dim=0)

        out = {
            'src_input_ids': src_input_ids,
            'src_length': src_length,
            'tgt_input_ids': tgt_input_ids,
            'tgt_length': tgt_length,
            'perturbation_vectors': perturbation_vectors,  # Include in the batch
            'perturbation': [d['perturbation'] for d in batch],
            'nperts': [d['nperts'] for d in batch],
            'tgt_counts_t1': tgt_counts,  # Now a tensor
            'tgt_size_factor_t1': tgt_size_factor,  # Now a tensor
        }
        
        out[f'tgt_counts'] = tgt_counts

        return out
    # ORIGINAL
#     def collate(self, batch):
#         # src
#         src_input_ids = [torch.tensor(d['src_dataset']['input_ids']) for d in batch]
        
#         # print(f'src_input_ids: {src_input_ids}')
        
#         src_length = torch.tensor([d['src_dataset']['length'] for d in batch])
#         model_input_size = torch.max(src_length)
#         src_input_ids = pad_tensor_list(
#             src_input_ids, self.max_len, self.pad_token_id, model_input_size
#         )
#         # if batch[0]['src_counts'] is not None:
#         #     if isinstance(batch[0]['src_counts'], csr_matrix):
#         #         src_counts = [torch.tensor(d['src_counts'].A) for d in batch]
#         #     else:
#         #         src_counts = [torch.tensor(d['src_counts']) for d in batch]
#         #     src_counts = torch.cat(src_counts, dim=0)
#         # else:
#         #     src_counts = None

#         # if self.condition_encodings:
#         #     condition = [d['conditions'] for d in batch]
#         #     condition_combined = torch.stack(
#         #         [d['conditions_combined'] for d in batch]
#         #         )
#         # else:
#         #     condition, condition_combined = None, None

#         tgt_counts_list = [d['tgt_counts'] for d in batch]
#         tgt_counts = torch.stack(tgt_counts_list, dim=0).float()  # Shape: (batch_size, n_genes)

#         conditions_combined = [1 for d in batch]
        
#         # tgt
#         tgt_input_ids = [torch.tensor(d['tgt_dataset']['input_ids']) for d in batch]
#         tgt_length = torch.tensor([d['tgt_dataset']['length'] for d in batch])
        
#         # print(f'tgt_input_ids: {tgt_input_ids}')
        
#         tgt_input_ids = pad_tensor_list(
#             tgt_input_ids, self.max_len, self.pad_token_id, model_input_size
#         )

# #         # Collect tgt_counts from the batch
# #         tgt_counts_list = [torch.tensor(d['tgt_counts']) for d in batch]
        
# #         # print(f'tgt_counts_list: {tgt_counts_list}')
        
# #         tgt_counts = pad_tensor_list(
# #             tgt_counts_list, self.max_len, self.pad_token_id, model_input_size
# #         )

# #         # Ensure tgt_counts is of correct shape and type
# #         tgt_counts = torch.stack(tgt_counts, dim=0).float()  # Shape: (batch_size, seq_len)

#     #         model_input_size = torch.max(tgt_length)
#         if batch[0]['tgt_counts'] is not None:
#             if isinstance(batch[0]['tgt_counts'], csr_matrix):
#                 tgt_counts_matrices = [d['tgt_counts'].A for d in batch]
#             else:
#                 tgt_counts_matrices = [d['tgt_counts'] for d in batch]

#             tgt_counts = [torch.tensor(matrix) for matrix in tgt_counts_matrices]
#             tgt_size_factor = [
#                 torch.tensor(np.ravel(matrix.sum(axis=1)))
#                 for matrix in tgt_counts_matrices
#             ]
#             tgt_counts = torch.cat(tgt_counts, dim=0)
#             tgt_size_factor = torch.cat(tgt_size_factor, dim=0)
#         else:
#             tgt_counts = None
#             tgt_size_factor = None
            
            
        # tgt_cell_idx = [d['tgt_dataset']['cell_pairing_index'] for d in batch]
        
#         # Gather perturbation embeddings
#         perturbed_embeddings_list = [d['perturbed_embeddings'] for d in batch]
#         perturbed_embeddings = torch.stack(perturbed_embeddings_list, dim=0) # Shape: (batch_size, max_nperts, embedding_dim)
        
        
        
#         out = {
#             'src_input_ids': src_input_ids,
#             'src_length': src_length,
#             'tgt_input_ids_t1': tgt_input_ids,
#             'tgt_length': tgt_length,
#             'tgt_cell_idx': tgt_cell_idx,
#             'combined_batch': [1 for d in batch],
#             'perturbation': [d['perturbation'] for d in batch],
#             'nperts': [d['nperts'] for d in batch],
#             'perturbed_embeddings': perturbed_embeddings,  # Add perturbation embeddings to the batch
#             'tgt_counts_t1': tgt_counts,  # Add true counts to the batch
#         }
        
#         print("tgt_input_ids_t1 shape:", out['tgt_input_ids_t1'].shape)
#         print("tgt_counts_t1 shape:", out['tgt_counts_t1'].shape)

#         # Handling other fields like 'disease', 'moe_one_hot_categories', etc.
#         # for var in self.var_list:
#         #     out[var] = [d['tgt_dataset'][var] for d in batch]
# #         if out['disease'] is not None:
# #             categories = out['disease']
# #             num_classes = len(set(categories))
# #             # Convert categories to indices and then to one-hot encoding
# #             category_indices = torch.tensor(
# #                 Categorical(categories).codes, dtype=torch.long
# #             )
# #             one_hot_labels = torch.eye(num_classes)[category_indices]
# #             out['moe_one_hot_categories'] = one_hot_labels
# #             out['moe_categories'] = Categorical(categories).categories
        
#         return out

# if __name__ == '__main__':
#     from datasets import load_from_disk

#     # test dataloader
#     # data_module = CellGenDataModule(
#     #     src_dataset=(
#     #         '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
#     #         'T_perturb/T_perturb/pp/res/dataset/'
#     #         'cytoimmgen_tokenised_degs_stratified_pairing_0h.dataset'
#     #     ),
#     #     tgt_datasets=(
#     #         '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
#     #         'T_perturb/T_perturb/pp/res/dataset/'
#     #         'cytoimmgen_tokenised_degs_stratified_pairing_16h.dataset'
#     #     ),
#     #     max_len=334,
#     # )
#     # data_module.setup()
#     # dataloader = data_module.train_dataloader()
#     # # iterate through batches
#     # train_iterator = iter(dataloader)
#     # batch = next(train_iterator)
#     # print(batch['tgt_input_ids'][:20, :20])
#     # print(len(batch['tgt_counts'][0]))
#     src_dataset = load_from_disk(
#         '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
#         't_generative/CellGen-reproducibility/covid_ipf_copd/'
#         'processed_data/dataset_hvg_src/control.dataset'
#     )
#     tgt_dataset = load_from_disk(
#         '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
#         't_generative/CellGen-reproducibility/covid_ipf_copd/'
#         'processed_data/dataset_hvg_tgt/disease.dataset'
#     )
#     with open(
#         '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
#         't_generative/CellGen-reproducibility/'
#         'covid_ipf_copd/processed_data/metadata.pkl',
#         'rb',
#     ) as f:
#         ipf_cell_pairing = pickle.load(f)

#     dataset = CellGenDataset(
#         src_dataset,
#         tgt_dataset,
#         ipf_cell_pairing,
#         split_indices=list(range(10)),
#     )
#     dataset[0]