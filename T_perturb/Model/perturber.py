from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from T_perturb.Model.trainer import CellGenTrainer
from T_perturb.src.utils import generate_pad, mean_nonpadding_embs


class CellGenPerturber(CellGenTrainer):
    def __init__(
        self,
        perturbation_mode: str = 'KO',
        perturbation_genes: List[str] = ['first_gene'],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.test_dict = self.test_dict | {
            'cls_cosine_similarities': [],
            'mean_cosine_similarities': [],
            'mean_gene_embeddings': [],
            'perturbed_cls_embeddings': [],
            'perturbed_gene_embeddings': [],
            'perturbed_mean_gene_embeddings': [],
            'perturbed_gene': [],
            'time_step': [],
            'tgt_lengths': [],
            'src_lengths': [],
        }

        # set perturbation information
        self.perturbation_mode = perturbation_mode
        self.perturbation_genes = perturbation_genes

    def move_gene_to_last(self, batch_tensor, indices):
        # Create a new tensor to store the result
        shifted_tensor = batch_tensor.clone()

        for i, j in indices:
            # Get the row
            row = batch_tensor[i]

            # Find the first padded index (assuming pad token is zero)
            first_zero_indices = (row == 0).nonzero(as_tuple=True)[0]
            if first_zero_indices.shape[0] > 0:
                first_zero_index = first_zero_indices[0]

                # Create a shifted version of the row
                shifted_tensor[i] = torch.cat(
                    [
                        row[:j],
                        row[j + 1 : first_zero_index],
                        row[j : j + 1],
                        row[first_zero_index:],
                    ]
                )
            else:
                # If no zero is found
                shifted_tensor[i] = torch.cat([row[:j], row[j + 1 :], row[j : j + 1]])

        return shifted_tensor

    def move_gene_to_50(self, batch_tensor, indices):
        # Create a new tensor to store the result
        shifted_tensor = batch_tensor.clone()

        for i, j in indices:
            # Get the row
            row = batch_tensor[i]

            # Find the first padded index (assuming pad token is zero)
            first_zero_indices = (row == 0).nonzero(as_tuple=True)[0]
            if first_zero_indices.shape[0] > 0 and first_zero_indices[0] < 100:
                first_zero_index = first_zero_indices[0]

                # Create a shifted version of the row
                shifted_tensor[i] = torch.cat(
                    [
                        row[:j],
                        row[j + 1 : first_zero_index],
                        row[j : j + 1],
                        row[first_zero_index:],
                    ]
                )
            elif (
                first_zero_indices.shape[0] > 0 and first_zero_indices[0] >= 100
            ) or row.shape[0] > 100:
                # Create a shifted version of the row
                shifted_tensor[i] = torch.cat(
                    [row[:j], row[j + 1 : 50], row[j : j + 1], row[50:]]
                )
            else:
                # If no zero is found
                shifted_tensor[i] = torch.cat([row[:j], row[j + 1 :], row[j : j + 1]])

        return shifted_tensor

    def move_gene_to_20(self, batch_tensor, indices):
        # Create a new tensor to store the result
        shifted_tensor = batch_tensor.clone()

        for i, j in indices:
            # Get the row
            row = batch_tensor[i]

            shifted_tensor[i] = torch.cat(
                [row[:j], row[j + 1 : 20], row[j : j + 1], row[20:]]
            )

        return shifted_tensor

    def move_gene_to_second_last(self, batch_tensor, indices):
        # Create a new tensor to store the result
        shifted_tensor = batch_tensor.clone()

        for i, j in indices:
            # Get the row
            row = batch_tensor[i]

            # Find the first padded index (assuming pad token is zero)
            first_zero_indices = (row == 0).nonzero(as_tuple=True)[0]
            if first_zero_indices.shape[0] > 0:
                first_zero_index = first_zero_indices[0]

                # Create a shifted version of the row
                shifted_tensor[i] = torch.cat(
                    [
                        row[:j],
                        row[j + 1 : first_zero_index - 5],
                        row[j : j + 1],
                        row[first_zero_index - 5 :],
                    ]
                )
            else:
                # If no zero is found
                shifted_tensor[i] = torch.cat(
                    [row[:j], row[j + 1 : -5], row[j : j + 1, row[-5:]]]
                )

        return shifted_tensor

    def move_gene_to_last_regardless(self, batch_tensor, indices):
        # Create a new tensor to store the result
        shifted_tensor = batch_tensor.clone()

        for i, j in indices:
            # Get the row
            row = batch_tensor[i]

            shifted_tensor[i] = torch.cat([row[:j], row[j + 1 :], row[j : j + 1]])

        return shifted_tensor

    def move_gene_to_first(self, batch_tensor, indices):
        # Create a new tensor to store the result
        shifted_tensor = batch_tensor.clone()

        for i, j in indices:
            # Get the row
            row = batch_tensor[i]

            shifted_tensor[i] = torch.cat([row[j : j + 1], row[:j], row[j + 1 :]])

        return shifted_tensor

    def perturb_mean_nonpadding_embs(self, batch, outputs, time_step):
        tgt_input_id = torch.cat(
            (
                getattr(self, f'cls_token_{str(time_step)}').expand(
                    batch[f'tgt_input_ids_t{time_step}'].shape[0], -1
                ),
                batch[f'tgt_input_ids_t{time_step}'],
            ),
            dim=1,
        )
        tgt_pad = generate_pad(tgt_input_id)
        mean_embedding = mean_nonpadding_embs(
            embs=outputs['dec_embedding'][time_step],
            pad=tgt_pad,
        )

        return mean_embedding

    def test_step(self, batch, *args, **kwargs):
        if self.perturbation_mode == 'KO':
            outputs = self.forward(batch)

            perturbation_genes_ids = {
                v: k
                for k, v in self.subset_tokenid_to_genename.items()
                if v in self.perturbation_genes
            }
            for gene in self.perturbation_genes:
                if gene != 'first_gene' and gene not in perturbation_genes_ids:
                    continue

                for time_step in self.time_steps:
                    # get unperturbed embeddings
                    dec_embedding = outputs['dec_embedding'][time_step]
                    cls_embeddings = dec_embedding[:, 0, :]
                    mean_gene_embeddings = outputs['mean_embedding'][time_step]

                    token_ids = batch[f'tgt_input_ids_t{time_step}']
                    cell_ids = batch[f'tgt_cell_idx_t{time_step}']

                    # perturb the gene
                    perturbed_batch = batch.copy()
                    perturbed_batch[f'tgt_input_ids_t{time_step}'] = batch[
                        f'tgt_input_ids_t{time_step}'
                    ].clone()
                    if gene == 'first_gene':
                        first_token_ids = batch[f'tgt_input_ids_t{time_step}'][:, 0]
                        # masking first gene
                        perturbed_batch[f'tgt_input_ids_t{time_step}'][
                            :, 0
                        ] = self.transformer.mask_token
                    else:
                        # masking the gene
                        perturbed_batch[f'tgt_input_ids_t{time_step}'][
                            token_ids == perturbation_genes_ids[gene]
                        ] = self.transformer.mask_token
                    perturbation_outputs = self.forward(perturbed_batch)

                    # get perturbation embeddings
                    pert_dec_embedding = perturbation_outputs['dec_embedding'][
                        time_step
                    ]
                    pert_cls_embeddings = pert_dec_embedding[:, 0, :]

                    # pad the perturbation gene to calculate the mean embeddings
                    if gene == 'first_gene':
                        perturbed_batch[f'tgt_input_ids_t{time_step}'][:, 0] = 0
                    else:
                        perturbed_batch[f'tgt_input_ids_t{time_step}'][
                            token_ids == perturbation_genes_ids[gene]
                        ] = 0
                    pert_mean_gene_embeddings = self.perturb_mean_nonpadding_embs(
                        perturbed_batch, perturbation_outputs, time_step
                    )

                    if gene == 'first_gene':
                        # Calculate cosine similarity for the filtered embeddings
                        cls_cosine_similarities = F.cosine_similarity(
                            cls_embeddings, pert_cls_embeddings, dim=1
                        )
                        mean_cosine_similarities = F.cosine_similarity(
                            mean_gene_embeddings, pert_mean_gene_embeddings, dim=1
                        )

                        # Store the results
                        self.test_dict['batch'].append(
                            batch['combined_batch'].detach().cpu()
                        )
                        self.test_dict['cell_idx'].append(cell_ids)
                        self.test_dict['true_counts'].append(
                            batch[f'tgt_counts_t{time_step}'].detach().cpu()
                        )
                        self.test_dict['cls_cosine_similarities'].append(
                            cls_cosine_similarities.detach().cpu()
                        )
                        self.test_dict['mean_cosine_similarities'].append(
                            mean_cosine_similarities.detach().cpu()
                        )
                        self.test_dict['cls_embeddings'].append(
                            cls_embeddings.detach().cpu()
                        )
                        self.test_dict['mean_gene_embeddings'].append(
                            mean_gene_embeddings.detach().cpu()
                        )
                        self.test_dict['perturbed_cls_embeddings'].append(
                            pert_cls_embeddings.detach().cpu()
                        )
                        self.test_dict['perturbed_mean_gene_embeddings'].append(
                            pert_mean_gene_embeddings.detach().cpu()
                        )
                        self.test_dict['perturbed_gene'].append(
                            [
                                self.subset_tokenid_to_genename[token_id.item()]
                                for token_id in first_token_ids
                            ]
                        )
                        self.test_dict['time_step'].append(
                            [time_step for i in range(cls_embeddings.size(0))]
                        )

                        # to delete later
                        self.test_dict['tgt_lengths'].append(
                            batch[f'tgt_length_t{time_step}'].detach().cpu()
                        )
                        self.test_dict['src_lengths'].append(
                            batch['src_length'].detach().cpu()
                        )

                        if len(self.var_list) > 0:
                            for var in self.var_list:
                                self.test_dict[var].append(batch[f'{var}_t{time_step}'])
                    else:
                        # Create a boolean mask for cells containing perturbed gene
                        cond_perturb = (token_ids == perturbation_genes_ids[gene]).sum(
                            1
                        ) > 0
                        cond_perturb_indx = torch.where(cond_perturb)[0]

                        # Calculate cosine similarity for the filtered embeddings
                        cls_cosine_similarities = F.cosine_similarity(
                            cls_embeddings[cond_perturb],
                            pert_cls_embeddings[cond_perturb],
                            dim=1,
                        )
                        mean_cosine_similarities = F.cosine_similarity(
                            mean_gene_embeddings[cond_perturb],
                            pert_mean_gene_embeddings[cond_perturb],
                            dim=1,
                        )

                        # Store the results
                        self.test_dict['batch'].append(
                            batch['combined_batch'][cond_perturb].detach().cpu()
                        )
                        self.test_dict['cell_idx'].append(
                            [cell_ids[i.item()] for i in cond_perturb_indx]
                        )
                        self.test_dict['true_counts'].append(
                            batch[f'tgt_counts_t{time_step}'][cond_perturb]
                            .detach()
                            .cpu()
                        )
                        self.test_dict['cls_cosine_similarities'].append(
                            cls_cosine_similarities.detach().cpu()
                        )
                        self.test_dict['mean_cosine_similarities'].append(
                            mean_cosine_similarities.detach().cpu()
                        )
                        self.test_dict['cls_embeddings'].append(
                            cls_embeddings[cond_perturb].detach().cpu()
                        )
                        # self.test_dict['gene_embeddings'].append(gene_embeddings[cond_perturb].detach().cpu())
                        self.test_dict['mean_gene_embeddings'].append(
                            mean_gene_embeddings[cond_perturb].detach().cpu()
                        )
                        self.test_dict['perturbed_cls_embeddings'].append(
                            pert_cls_embeddings[cond_perturb].detach().cpu()
                        )
                        # self.test_dict['perturbed_gene_embeddings'].append(pert_gene_embeddings[cond_perturb].detach().cpu())
                        self.test_dict['perturbed_mean_gene_embeddings'].append(
                            pert_mean_gene_embeddings[cond_perturb].detach().cpu()
                        )
                        self.test_dict['perturbed_gene'].append(
                            [gene for i in cond_perturb_indx]
                        )
                        self.test_dict['time_step'].append(
                            [time_step for i in cond_perturb_indx]
                        )

                        # to delete later
                        self.test_dict['tgt_lengths'].append(
                            batch[f'tgt_length_t{time_step}'][cond_perturb]
                            .detach()
                            .cpu()
                        )
                        self.test_dict['src_lengths'].append(
                            batch['src_length'][cond_perturb].detach().cpu()
                        )

                        if len(self.var_list) > 0:
                            for var in self.var_list:
                                self.test_dict[var].append(
                                    [
                                        batch[f'{var}_t{time_step}'][i.item()]
                                        for i in cond_perturb_indx
                                    ]
                                )

    def on_test_epoch_end(self):
        print('Start saving embeddings -------------------')
        cls_embeddings = torch.cat(self.test_dict['cls_embeddings'])
        true_counts = torch.cat(self.test_dict['true_counts'])
        batch = torch.cat(self.test_dict['batch'])
        cell_ids = np.concatenate(self.test_dict['cell_idx'])
        # gene_embeddings = torch.cat(self.test_dict['gene_embeddings'])
        if len(self.var_list) > 0:
            var_dict = {}
            for var in self.var_list:
                var_dict[var] = np.concatenate(self.test_dict[var])
            test_obs = pd.DataFrame(var_dict)
        else:
            test_obs = pd.DataFrame()
        test_obs['batch'] = np.array(batch)
        test_obs['cell_idx'] = cell_ids

        if self.perturbation_mode == 'KO':
            cls_cosine_similarities = torch.cat(
                self.test_dict['cls_cosine_similarities']
            )
            mean_cosine_similarities = torch.cat(
                self.test_dict['mean_cosine_similarities']
            )
            mean_gene_embeddings = torch.cat(self.test_dict['mean_gene_embeddings'])
            perturbed_cls_embeddings = torch.cat(
                self.test_dict['perturbed_cls_embeddings']
            )
            perturbed_mean_gene_embeddings = torch.cat(
                self.test_dict['perturbed_mean_gene_embeddings']
            )
            perturbed_gene = np.concatenate(self.test_dict['perturbed_gene'])
            time_step = np.concatenate(self.test_dict['time_step'])
            tgt_lengths = np.concatenate(self.test_dict['tgt_lengths'])
            src_lengths = np.concatenate(self.test_dict['src_lengths'])

            test_obs['cls_cosine_similarities'] = cls_cosine_similarities.numpy()
            test_obs['mean_cosine_similarities'] = mean_cosine_similarities.numpy()
            test_obs['gene'] = perturbed_gene
            test_obs['time_step'] = time_step
            test_obs['tgt_lengths'] = tgt_lengths
            test_obs['src_lengths'] = src_lengths

            perturbed_mean_gene_embeddings_np = perturbed_mean_gene_embeddings.numpy()
            adata = ad.AnnData(
                X=true_counts.numpy(),
                obs=test_obs,
                obsm={
                    'cls_embeddings': cls_embeddings.numpy(),
                    # 'gene_embeddings': gene_embeddings.numpy(),
                    'mean_gene_embeddings': mean_gene_embeddings.numpy(),
                    'perturbed_cls_embeddings': perturbed_cls_embeddings.numpy(),
                    # 'perturbed_gene_embeddings': perturbed_gene_embeddings.numpy(),
                    'perturbed_mean_gene_embeddings': perturbed_mean_gene_embeddings_np,
                },
            )

        if self.gene_names is not None:
            adata.var_names = self.gene_names
        # save anndata
        adata.write_h5ad(
            f'{self.output_dir}/{self.date}_'
            f'perturbation_embeddings_cosine_similarity.h5ad'
        )
        print('End saving embeddings -------------------')
