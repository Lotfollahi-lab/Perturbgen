import pickle
import random
from typing import (
    Any,
    List,
    Literal,
)

import evaluate
import numpy as np
import pandas as pd
import torch

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CellGenTrainer
from T_perturb.Perturb.T_model import PerturberMasking
from T_perturb.src.optimal_transport import wasserstein
from T_perturb.src.utils import (
    compute_rouge_score,
    map_results_to_genes,
    return_gene_embeddings,
    return_perturbation_adata,
)


class PerturberTrainer(CellGenTrainer):
    def __init__(
        self,
        generate: bool = False,
        sequence_length: int = 2048,
        temperature: float = 2.0,
        iterations: int = 18,
        mapping_dict_path: str | None = None,
        genes_to_perturb: List[str] | None = None,
        validation_mode: Literal['inference', 'generate'] | None = None,
        perturbation_mode: Literal['mask', 'pad', 'delete', 'overexpress']
        | None = None,
        perturbation_sequence: Literal['src', 'tgt'] | None = None,
        gene_module_list: List[str] | None = None,
        num_of_background_genes: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transformer = PerturberMasking(
            tgt_vocab_size=kwargs['tgt_vocab_size'],
            d_model=kwargs['d_model'],
            num_heads=kwargs['num_heads'],
            num_layers=kwargs['num_layers'],
            d_ff=kwargs['d_ff'],
            max_seq_length=kwargs['max_seq_length'],
            dropout=kwargs['dropout'],
            pred_tps=kwargs['pred_tps'],
            context_tps=kwargs['context_tps'] if 'context_tps' in kwargs else None,
            n_total_tps=kwargs['n_total_tps'],
            encoder=kwargs['encoder'],
            mask_scheduler=kwargs['mask_scheduler'],
            pos_encoding_mode=kwargs['pos_encoding_mode'],
            return_attn=kwargs['return_attn'],
            context_mode=kwargs['context_mode'],
        )
        self.validation_mode = validation_mode
        if validation_mode is not None:
            if perturbation_sequence is None:
                raise ValueError(
                    'Please specify the perturbation_sequence: "src" or "tgt"'
                )
            if perturbation_mode is None:
                raise ValueError('Please specify the perturbation_token')
            if genes_to_perturb is None:
                raise ValueError('Please specify the genes_to_perturb')
            self.perturbation_sequence = perturbation_sequence
            self.genes_to_perturb = genes_to_perturb
            if mapping_dict_path is not None:
                with open(
                    mapping_dict_path,
                    'rb',
                ) as f:
                    tokenid_to_gene = pickle.load(f)
                gene_to_token_id = {v: k for k, v in tokenid_to_gene.items()}
                self.gene_to_tokenid = gene_to_token_id
                # find corresponding special for dictionary keys '<>'
                special_tokens = [
                    k for k, v in tokenid_to_gene.items() if v.startswith('<')
                ]
                self.special_tokens = special_tokens
                tokens_to_perturb = [
                    gene_to_token_id[gene] for gene in self.genes_to_perturb
                ]
                self.tokens_to_perturb = torch.tensor(
                    tokens_to_perturb, dtype=torch.long, device=self.device
                )
            else:
                self.tokens_to_perturb = torch.tensor(
                    genes_to_perturb, dtype=torch.long, device=self.device
                )
            self.perturbation_mode = perturbation_mode

            print(
                f'Start perturbation ...\n'
                f'- Validation mode: {self.validation_mode}\n'
                f'- Perturbation sequence: {self.perturbation_sequence}\n'
                f'- Perturbing genes: {genes_to_perturb}\n'
                f'- Perturbation mode: {perturbation_mode}\n'
            )
            if gene_module_list is not None:
                self.gene_module_list: List[str] | None = gene_module_list
                if num_of_background_genes is None:
                    raise ValueError('Please specify the number of background genes')
                # exclude special tokens from self.gene_to_tokenid
                gene_tokens_filtered = {
                    gene: token
                    for gene, token in self.gene_to_tokenid.items()
                    if gene not in self.special_tokens
                }
                # exclude perturbation tokens from gene_tokens_filtered
                gene_tokens_filtered = {
                    gene: token
                    for gene, token in gene_tokens_filtered.items()
                    if token not in self.tokens_to_perturb
                }
                self.gene_module_dict = {
                    gene: gene_tokens_filtered[gene] for gene in gene_module_list
                }

                # remove gene_module_tokens from selection of background genes
                background_gene_dict = {
                    gene: token
                    for gene, token in gene_tokens_filtered.items()
                    if gene not in gene_module_list
                }
                # filter out all values which are >100 in background_gene_dict
                background_gene_dict = {
                    gene: token
                    for gene, token in background_gene_dict.items()
                    if token < 100
                }
                random_entries = random.sample(
                    list(background_gene_dict.items()), num_of_background_genes
                )
                self.background_gene_dict = dict(random_entries)
            else:
                self.gene_module_list = None

        self.generate = generate
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.iterations = iterations

        for key in [
            'cls_cosine_similarity',
            'mean_cosine_similarity',
            'gene_cosine_similarity',
            'delta_probs',
            'delta_gene_probs',
            'wasserstein_distance',
            'true_cls',
            'perturbed_cls',
            'true_gm_embs',
            'perturbed_gm_embs',
            'true_background_embs',
            'perturbed_background_embs',
        ]:
            self.test_dict[key] = []

        self.rouge = evaluate.load('rouge')
        self.rouge_seq_len_list = [25, 100, kwargs['max_seq_length']]
        for seq_len in self.rouge_seq_len_list:
            self.test_dict[f'rouge1_{seq_len}'] = []
        self.max_seq_length = kwargs['max_seq_length']
        self.encoder = kwargs['encoder']
        self.pos_encoding_mode = kwargs['pos_encoding_mode']
        self.mask_scheduler = kwargs['mask_scheduler']

    def forward(
        self,
        batch: Any,
        perturbation: bool = False,
    ):
        if perturbation:
            self.tokens_to_perturb = self.tokens_to_perturb.to(self.device)
            if self.perturbation_mode in ['mask', 'pad']:
                if self.perturbation_mode == 'mask':
                    perturbation_token = self.gene_to_tokenid['<mask>']
                elif self.perturbation_mode == 'pad':
                    perturbation_token = self.gene_to_tokenid['<pad>']
                self.perturbation_token = torch.tensor(
                    perturbation_token, dtype=torch.long, device=self.device
                )
        tgt_input_id_dict = {}
        for i in self.pred_tps:
            tgt_input_id_ = batch[f'tgt_input_ids_t{i}'].clone()
            if perturbation:
                if (
                    self.perturbation_sequence is not None
                    and 'tgt' in self.perturbation_sequence
                ):
                    perturbed_tgt = tgt_input_id_.clone()
                    mask = torch.isin(tgt_input_id_, self.tokens_to_perturb)
                    if self.perturbation_mode in ['delete', 'overexpress']:
                        # Create a mask for elements not equal to the target token
                        mask = perturbed_tgt != self.tokens_to_perturb
                        # add padding mask
                        pad_mask = perturbed_tgt != self.pad_token_id
                        mask_ = mask & pad_mask
                        # Count the number of valid tokens in each sequence
                        valid_counts = mask_.sum(dim=1)
                        # Get indices for valid tokens
                        valid_tokens = perturbed_tgt[mask_]

                        # Initialize the result tensor filled with pad_token_id
                        perturbed_tgt = torch.full_like(
                            perturbed_tgt, self.pad_token_id
                        )

                        # Use advanced indexing to fill the valid tokens
                        # into the perturbed_tgt tensor
                        batch_indices = torch.arange(
                            perturbed_tgt.size(0), device=perturbed_tgt.device
                        ).repeat_interleave(valid_counts)
                        position_indices = torch.cat(
                            [
                                torch.arange(c, device=perturbed_tgt.device)
                                for c in valid_counts
                            ]
                        )
                        perturbed_tgt[batch_indices, position_indices] = valid_tokens
                    else:
                        perturbed_tgt[mask] = self.perturbation_token
                    tgt_input_id_ = perturbed_tgt.clone()
                    # add another if to concatenate overexpressed genes
                    if self.perturbation_mode == 'overexpress':
                        # concatenate tokens_to_perturb to tgt_input_id_
                        tgt_input_id_ = torch.cat(
                            (
                                self.tokens_to_perturb.expand(
                                    tgt_input_id_.shape[0], -1
                                ),
                                tgt_input_id_,
                            ),
                            dim=1,
                        )

            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        tgt_input_id_.shape[0], -1
                    ),
                    tgt_input_id_,
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if perturbation:
            if (
                self.perturbation_sequence is not None
                and 'src' in self.perturbation_sequence
            ):
                perturbed_src = batch['src_input_ids'].clone()
                if self.perturbation_mode == 'delete':
                    # Create a mask for elements not equal to the target token
                    mask = perturbed_src != self.tokens_to_perturb
                    # add padding mask
                    pad_mask = perturbed_src != self.pad_token_id
                    mask_ = mask & pad_mask
                    # Count the number of valid tokens in each sequence
                    valid_counts = mask_.sum(dim=1)
                    # Get indices for valid tokens
                    valid_tokens = perturbed_src[mask_]

                    # Initialize the result tensor filled with pad_token_id
                    perturbed_src = torch.full_like(perturbed_src, self.pad_token_id)

                    # Use advanced indexing to fill the valid tokens
                    # into the perturbed_src tensor
                    batch_indices = torch.arange(
                        perturbed_src.size(0), device=perturbed_src.device
                    ).repeat_interleave(valid_counts)
                    position_indices = torch.cat(
                        [
                            torch.arange(c, device=perturbed_src.device)
                            for c in valid_counts
                        ]
                    )
                    perturbed_src[batch_indices, position_indices] = valid_tokens
                else:
                    mask = torch.isin(perturbed_src, self.tokens_to_perturb)
                    perturbed_src[mask] = self.perturbation_token
            else:
                perturbed_src = batch['src_input_ids']
        else:
            perturbed_src = batch['src_input_ids']

        if self.validation_mode == 'inference':
            outputs = self.transformer(
                src_input_id=perturbed_src,
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=True,
            )
        else:
            outputs = self.transformer.forward(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=self.return_embeddings,
            )

        return outputs, perturbed_src, tgt_input_id_dict

    def mean_perturbation_embs(
        self,
        embs: torch.Tensor,
        input_ids: torch.Tensor,
        mapping_dict: dict,
        perturbation_tokens: torch.Tensor,
        dim: int = 1,
    ):
        '''
        Compute the mean of the non-padding embeddings.
        Modified from Geneformer:
        https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py # noqa
        Accessed: 2024-05-14
        '''

        # create a mask to exclude special and perturbation tokens
        special_token_names = ['<cls>', '<mask>', '<pad>', '<eos>']
        special_tokens = [mapping_dict[token] for token in special_token_names]
        tokens_to_exclude = torch.cat(
            (
                torch.tensor(special_tokens, device=perturbation_tokens.device),
                perturbation_tokens,
            )
        )
        pad_mask = torch.isin(input_ids, tokens_to_exclude)
        # create a tensor of original lengths
        original_lens = pad_mask.sum(dim=1)

        # create CLS token mask
        if embs.dim() == 3:
            # fill the masked positions in embs with zeros
            masked_embs = embs.masked_fill(pad_mask.unsqueeze(2), 0.0)
            # compute the mean across the non-padding dimensions
            mean_embs = masked_embs.sum(dim) / original_lens.view(-1, 1).float()

        elif embs.dim() == 2:
            masked_embs = embs.masked_fill(pad_mask, 0.0)
            mean_embs = masked_embs.sum(dim) / original_lens.float()
        return mean_embs

    def test_step(self, batch, *args, **kwargs):
        print(batch)
        raise
        if self.validation_mode == 'inference':
            true_outputs, _, _ = self.forward(batch, perturbation=False)
            perturbed_outputs, _, _ = self.forward(batch, perturbation=True)

        elif self.validation_mode == 'generate':
            # print(self.transformer)
            # self.transformer = self.quantize_model(self.transformer)
            (
                _,
                pert_src_input_ids,
                tgt_input_id_dict,
            ) = self.forward(batch, perturbation=True)
            decoder_kwargs = {
                'tgt_input_id_dict': tgt_input_id_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
            }

            true_outputs, true_ids_dict = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                genes_to_perturb=self.tokens_to_perturb,
                **decoder_kwargs,
            )
            perturbed_outputs, perturbed_ids_dict = self.transformer.generate(
                src_input_id=pert_src_input_ids,
                **decoder_kwargs,
            )
            for t in self.pred_tps:
                # pert_ids = perturbed_ids_dict[t].detach().cpu().numpy()
                true_ids = true_ids_dict[t].detach().cpu().numpy()
                # ground truth
                input_ids = batch[f'tgt_input_ids_t{t}'].detach().cpu().numpy()

                test_dict = compute_rouge_score(
                    rouge=self.rouge,
                    pred_ids=true_ids,
                    tgt_ids=input_ids,
                    rouge_len_list=self.rouge_seq_len_list,
                    max_seq_length=self.max_seq_length,
                    test_dict=self.test_dict,
                )
                self.test_dict = test_dict
        else:
            raise ValueError(
                f'Invalid perturbation mode: {self.validation_mode}:'
                f'Choose between "inference" or "generate"'
            )
        for t in self.pred_tps:
            # create a mask for special tokens to exlude
            # them from the cosine similarity & logits and probs

            true_cls = true_outputs[t]['dec_embedding'][:, 0, :]
            true_gene = true_outputs[t]['dec_embedding'][:, 1:, :]
            print(true_gene.shape)

            # true_logits = true_outputs[t]['dec_logits'][:, 1:, :]

            perturbed_cls = perturbed_outputs[t]['dec_embedding'][:, 0, :]
            perturbed_gene = perturbed_outputs[t]['dec_embedding'][:, 1:, :]
            print(perturbed_gene.shape)
            raise
            # perturbed_logits = perturbed_outputs[t]['dec_logits'][:, 1:, :]

            # true_probs = torch.softmax(true_logits, dim=-1)
            # perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
            # delta_probs = torch.abs(true_probs - perturbed_probs)
            # token_probs_change = delta_probs.sum(dim=-1)
            # delta_gene_probs, self.marker_genes = map_results_to_genes(
            #     token_probs_change,
            #     mapping_dict=self.gene_to_tokenid,
            #     token_ids=batch[f'tgt_input_ids_t{t}'],
            # )

            gene_cos_sim = cosine_similarity(
                true_gene,
                perturbed_gene,
                dim=-1,
            )
            gene_cos_sim, self.marker_genes = map_results_to_genes(
                gene_cos_sim,
                mapping_dict=self.gene_to_tokenid,
                token_ids=batch[f'tgt_input_ids_t{t}'],
            )

            cls_cos_sim = cosine_similarity(
                perturbed_cls,
                true_cls,
            )
            true_mean_embs = self.mean_perturbation_embs(
                true_gene,  # exclude cls token
                batch[f'tgt_input_ids_t{t}'],
                self.gene_to_tokenid,
                self.tokens_to_perturb,
                dim=1,
            )
            perturbed_mean_cls = self.mean_perturbation_embs(
                perturbed_gene,  # exclude cls token
                batch[f'tgt_input_ids_t{t}'],
                self.gene_to_tokenid,
                self.tokens_to_perturb,
                dim=1,
            )
            mean_cos_sim = cosine_similarity(
                perturbed_mean_cls,
                true_mean_embs,
            )

            # # iterate over the batch and compute the wasserstein distance
            # wd = []
            # for i in range(true_cls.shape[0]):
            #     print(true_cls[i].shape)
            #     wd.append(wasserstein(
            #         true_cls[i],
            #         perturbed_cls[i],
            #         power=1
            #     ))

            cls_cos_sim = cls_cos_sim.detach().cpu().to(torch.float16)
            mean_cos_sim = mean_cos_sim.detach().cpu().to(torch.float16)
            gene_cos_sim = gene_cos_sim.detach().cpu().to(torch.float16)
            true_cls = true_cls.detach().cpu().to(torch.float16)
            perturbed_cls = perturbed_cls.detach().cpu().to(torch.float16)

            # token_probs_change = token_probs_change.detach().cpu().to(torch.float16)
            # delta_probs = delta_probs.detach().cpu().to(torch.float16)
            # delta_gene_probs = delta_gene_probs.detach().cpu().to(torch.float16)
            self.test_dict['cls_cosine_similarity'].append(cls_cos_sim)
            self.test_dict['mean_cosine_similarity'].append(mean_cos_sim)
            self.test_dict['gene_cosine_similarity'].append(gene_cos_sim)
            self.test_dict['true_cls'].append(true_cls)
            self.test_dict['perturbed_cls'].append(perturbed_cls)

            if self.gene_module_list is not None:
                true_gm_embs = return_gene_embeddings(
                    true_gene,
                    self.gene_module_dict,
                    batch[f'tgt_input_ids_t{t}'],
                )
                perturbed_gm_embs = return_gene_embeddings(
                    perturbed_gene,
                    self.gene_module_dict,
                    batch[f'tgt_input_ids_t{t}'],
                )
                true_background_embs = return_gene_embeddings(
                    true_gene,
                    self.background_gene_dict,
                    batch[f'tgt_input_ids_t{t}'],
                )
                perturbed_background_embs = return_gene_embeddings(
                    perturbed_gene,
                    self.background_gene_dict,
                    batch[f'tgt_input_ids_t{t}'],
                )
                # convert float32 to calculate wasserstein distance
                true_gm_embs = true_gm_embs.detach().cpu().to(torch.float32)
                perturbed_gm_embs = perturbed_gm_embs.detach().cpu().to(torch.float32)
                true_background_embs = (
                    true_background_embs.detach().cpu().to(torch.float32)
                )
                perturbed_background_embs = (
                    perturbed_background_embs.detach().cpu().to(torch.float32)
                )
                self.test_dict['true_gm_embs'].append(true_gm_embs)
                self.test_dict['perturbed_gm_embs'].append(perturbed_gm_embs)
                self.test_dict['true_background_embs'].append(true_background_embs)
                self.test_dict['perturbed_background_embs'].append(
                    perturbed_background_embs
                )
            # return obs_key
            self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{t}'])
            if len(self.var_list) > 0:
                for var in self.var_list:
                    self.test_dict[var].append(batch[f'{var}_t{t}'])

    def compute_non_zero_mean(self, embs: torch.Tensor):
        non_zero_mask = embs != 0
        non_zero_sum = torch.sum(embs * non_zero_mask, dim=0)
        non_zero_count = torch.sum(non_zero_mask, dim=0)
        if torch.any(non_zero_count == 0):
            raise ValueError(
                'The embeddings contain positions where all values are zero.'
            )
        else:
            non_zero_mean = non_zero_sum / non_zero_count
        return non_zero_mean

    def on_test_epoch_end(self):
        # compute emd of gm_embs based on condition
        if self.gene_module_list is not None:
            true_gm_embs = torch.cat(self.test_dict['true_gm_embs'], dim=0)
            perturbed_gm_embs = torch.cat(self.test_dict['perturbed_gm_embs'], dim=0)
            true_background_embs = torch.cat(
                self.test_dict['true_background_embs'], dim=0
            )
            perturbed_background_embs = torch.cat(
                self.test_dict['perturbed_background_embs'], dim=0
            )
            # compute the wasserstein distance per condition
            gm_wd = {}
            background_wd = {}
            conditions = np.concatenate(self.test_dict['cell_type'])
            for condition in np.unique(conditions):
                true_gm_embs_cond = true_gm_embs[conditions == condition]
                perturbed_gm_embs_cond = perturbed_gm_embs[conditions == condition]
                true_background_embs_cond = true_background_embs[
                    conditions == condition
                ]
                perturbed_background_embs_cond = perturbed_background_embs[
                    conditions == condition
                ]
                # non-zero mean aggregation
                true_gm_embs_cond = self.compute_non_zero_mean(true_gm_embs_cond)
                perturbed_gm_embs_cond = self.compute_non_zero_mean(
                    perturbed_gm_embs_cond
                )
                true_background_embs_cond = self.compute_non_zero_mean(
                    true_background_embs_cond
                )
                perturbed_background_embs_cond = self.compute_non_zero_mean(
                    perturbed_background_embs_cond
                )
                # compute the wasserstein distance
                gm_wd[condition] = wasserstein(
                    true_gm_embs_cond,
                    perturbed_gm_embs_cond,
                )
                background_wd[condition] = wasserstein(
                    true_background_embs_cond,
                    perturbed_background_embs_cond,
                )
                # store results as dataframes and merge them
                gm_wd_df = pd.DataFrame.from_dict(
                    gm_wd, orient='index', columns=['gm_wd']
                )
                background_wd_df = pd.DataFrame.from_dict(
                    background_wd, orient='index', columns=['background_wd']
                )
                wd_df = pd.concat([gm_wd_df, background_wd_df], axis=1)
                # plot the results
                wd_df.plot(kind='bar')
                wd_df.to_csv(f'{self.output_dir}/wasserstein_distance.csv')

                del (
                    true_gm_embs_cond,
                    perturbed_gm_embs_cond,
                    true_background_embs_cond,
                    perturbed_background_embs_cond,
                )

        obs_key = self.var_list if len(self.var_list) > 0 else []
        obs_key.extend(['cell_idx'])
        if len(self.genes_to_perturb) > 1:
            genes_to_perturb = '_'.join(self.genes_to_perturb)
        else:
            genes_to_perturb = self.genes_to_perturb[0]
        if len(self.perturbation_sequence) > 1:
            perturbation_sequence = '_'.join(self.perturbation_sequence)
        else:
            perturbation_sequence = self.perturbation_sequence[0]
        return_perturbation_adata(
            test_dict=self.test_dict,
            obs_key=obs_key,
            output_dir=self.output_dir,
            marker_genes=self.marker_genes,
            file_name=(
                f'{self.date}_m{self.validation_mode}_adata'
                f'_g{genes_to_perturb}'
                f'_s{perturbation_sequence}'
                f'_t{self.perturbation_mode}.h5ad'
            ),
            mode=self.validation_mode,
        )
