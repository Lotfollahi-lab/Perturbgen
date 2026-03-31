import numpy as np
import torch
from datasets import Dataset
from typing import Optional


def dummy_dataset(
    max_len: int = 50,
    vocab_size: int = 100,
    num_samples: int = 100,
    total_time_steps: Optional[int] = None,
):
    if total_time_steps is None:
        # Generate unique indices for each sample using NumPy
        input_ids_np = np.array(
            [
                np.random.choice(np.arange(2, vocab_size), max_len, replace=False)
                for _ in range(num_samples)
            ]
        )
        input_ids = torch.tensor(input_ids_np, dtype=torch.long)
        input_ids[:, -10:] = 0
        cell_idx = np.arange(num_samples)
        cell_type = np.random.choice(['A', 'B', 'C'], num_samples)
        dataset = Dataset.from_dict(
            {
                'input_ids': input_ids,
                'cell_type': cell_type,
                'length': [len(input_ids)] * num_samples,
                'cell_pairing_index': cell_idx,
            }
        )
        return dataset
    else:
        tgt_dataset_dict = {}
        for t in range(total_time_steps):
            input_ids_np = np.array(
                [
                    np.random.choice(np.arange(2, vocab_size), max_len, replace=False)
                    for _ in range(num_samples)
                ]
            )
            input_ids = torch.tensor(input_ids_np, dtype=torch.long)
            input_ids[:, -10:] = 0
            tgt_dataset_dict[f'tgt_dataset_t{t+1}'] = Dataset.from_dict(
                {
                    'input_ids': input_ids,
                    'cell_type': np.random.choice(['A', 'B', 'C'], num_samples),
                    'length': [len(input_ids)] * num_samples,
                    'cell_pairing_index': np.random.choice(
                        100, num_samples, replace=False
                    ),
                }
            )
        return tgt_dataset_dict
# create cell x gene matrix with 100 cells and 100 genes
def dummy_cell_gene_matrix(
    num_cells: int = 100,
    num_genes: int = 100,
    total_time_steps: Optional[int] = None,
):
    lambda_param = 10
    if total_time_steps is None:
        gex_matrix = np.random.poisson(lambda_param, (num_cells, num_genes))
        gex_matrix = gex_matrix.astype(np.float32)
        return np.expand_dims(gex_matrix, axis=1)
    else:
        counts_dict = {}
        for t in range(total_time_steps):
            gex_matrix = np.random.poisson(lambda_param, (num_cells, num_genes))
            gex_matrix = gex_matrix.astype(np.float32)
            counts_dict[f'tgt_h5ad_t{t+1}'] = np.expand_dims(gex_matrix, axis=1)
        return counts_dict
