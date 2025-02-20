import pickle
from typing import (
    Any,
    List,
    Literal,
)

import evaluate
import torch

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CytoMeisterTrainer
from T_perturb.Perturb.T_model import PerturberMasking
from T_perturb.src.utils import (
    compute_rouge_score,
    concat_cond_tokens,
    map_results_to_genes,
    mean_nonpadding_embs,
    return_perturbation_adata,
)


class PerturberTrainer(CytoMeisterTrainer):
    def __init__(
        self,
        generate: bool = False,
        sequence_length: int = 2048,
        temperature: float = 2.0,
        iterations: int = 18,
        mapping_dict_path: str | None = None,
        genes_to_perturb: List[str] | None = None,
        src_tokens_to_perturb: List[int] | None = None,
        tgt_tokens_to_perturb: List[int] | None = None,
        validation_mode: Literal['inference', 'generate'] | None = None,
        perturbation_mode: Literal['mask', 'pad', 'delete', 'overexpress']
        | None = None,
        perturbation_sequence: Literal['src', 'tgt'] | None = None,
        tokenid_to_rowid_path: str | None = None,
        # gene_module_list: List[str] | None = None,
        # num_of_background_genes: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.validation_mode = validation_mode
        if validation_mode is not None:
            if perturbation_sequence is None:
                raise ValueError(
                    'Please specify the perturbation_sequence: "src" or "tgt"'
                )
            if perturbation_mode is None:
                raise ValueError('Please specify the perturbation_token')

            self.perturbation_sequence = perturbation_sequence
            self.perturbation_mode = perturbation_mode

            # if gene_module_list is not None:
            #     self.gene_module_list: List[str] | None = gene_module_list
            #     if num_of_background_genes is None:
            #         raise ValueError('Please specify the number of background genes')
            #     # exclude special tokens from self.gene_to_tgtid
            #     gene_tokens_filtered = {
            #         gene: token
            #         for gene, token in self.gene_to_tgtid.items()
            #         if gene not in self.special_tokens
            #     }
            #     # exclude perturbation tokens from gene_tokens_filtered
            #     gene_tokens_filtered = {
            #         gene: token
            #         for gene, token in gene_tokens_filtered.items()
            #         if token not in self.tgt_pert_tokens
            #     }
            #     self.gene_module_dict = {
            #         gene: gene_tokens_filtered[gene] for gene in gene_module_list
            #     }

            #     # remove gene_module_tokens from selection of background genes
            #     background_gene_dict = {
            #         gene: token
            #         for gene, token in gene_tokens_filtered.items()
            #         if gene not in gene_module_list
            #     }
            #     # filter out all values which are >100 in background_gene_dict
            #     background_gene_dict = {
            #         gene: token
            #         for gene, token in background_gene_dict.items()
            #         if token < 100
            #     }
            #     random_entries = random.sample(
            #         list(background_gene_dict.items()), num_of_background_genes
            #     )
            #     self.background_gene_dict = dict(random_entries)
            # else:
            #     self.gene_module_list = None
        # dictionary to map gene names to row ids in tgt
        if mapping_dict_path is not None:
            with open(
                mapping_dict_path,
                'rb',
            ) as f:
                rowid_to_gene = pickle.load(f)
            gene_to_rowid = {v: k for k, v in rowid_to_gene.items()}
            self.gene_to_tgtid = gene_to_rowid

            # find corresponding special for dictionary keys '<>'
            special_tokens = [k for k, v in rowid_to_gene.items() if v.startswith('<')]
            self.special_tokens = special_tokens
        # dictionary to map gene names to row ids in src
        if tokenid_to_rowid_path is not None:
            with open(
                tokenid_to_rowid_path,
                'rb',
            ) as f:
                tokenid_to_rowid = pickle.load(f)
                rowid_to_tokenid = {v: k for k, v in tokenid_to_rowid.items()}
                # map back rowid to original tokenid
                gene_to_srcid = {
                    k: rowid_to_tokenid[gene_to_rowid[k]]
                    for k in gene_to_rowid
                    if gene_to_rowid[k] in rowid_to_tokenid
                }
        else:
            gene_to_srcid = None
        if perturbation_sequence is not None:
            if 'src' in perturbation_sequence:
                tgt_pert_tokens = None
                self.tgt_pert_tokens = tgt_pert_tokens
                if genes_to_perturb is not None:
                    if gene_to_srcid is not None:
                        src_pert_tokens = [
                            gene_to_srcid[gene] for gene in genes_to_perturb
                        ]
                        src_pert_tokens = torch.tensor(
                            src_pert_tokens, dtype=torch.long
                        )
                    else:
                        raise ValueError(
                            'Please specify the tokenid_to_rowid_path path '
                            'to map the perturbation token'
                        )
                elif src_tokens_to_perturb is not None:
                    src_pert_tokens = torch.tensor(
                        src_tokens_to_perturb, dtype=torch.long
                    )
                else:
                    raise ValueError(
                        (
                            'Please specify either genes_to_perturb'
                            'or src_tokens_to_perturb'
                        )
                    )
                self.register_buffer(
                    'src_pert_tokens', src_pert_tokens, persistent=False
                )
            elif 'tgt' in perturbation_sequence:
                if genes_to_perturb is not None:
                    tgt_pert_tokens = [gene_to_rowid[gene] for gene in genes_to_perturb]
                    tgt_pert_tokens = torch.tensor(tgt_pert_tokens, dtype=torch.long)
                elif tgt_tokens_to_perturb is not None:
                    tgt_pert_tokens = torch.tensor(
                        tgt_tokens_to_perturb, dtype=torch.long
                    )
                else:
                    raise ValueError(
                        (
                            'Please specify either genes_to_perturb'
                            'or tgt_tokens_to_perturb'
                        )
                    )
                self.register_buffer(
                    'tgt_pert_tokens', tgt_pert_tokens, persistent=False
                )
            self.genes_to_perturb = genes_to_perturb
        else:
            raise ValueError('Please specify the perturbation_sequence: "src" or "tgt"')
        print(
            f'Start perturbation ...\n'
            f'- Validation mode: {self.validation_mode}\n'
            f'- Perturbation sequence: {self.perturbation_sequence}\n'
            f'- Perturbing genes: {genes_to_perturb}\n'
            f'- Perturbation mode: {perturbation_mode}\n'
        )
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
            encoder_path=kwargs['encoder_path'],
            mask_scheduler=kwargs['mask_scheduler'],
            pos_encoding_mode=kwargs['pos_encoding_mode'],
            return_attn=kwargs['return_attn'],
            context_mode=kwargs['context_mode'],
            condition_dict=kwargs['condition_dict'],
            gene_to_rowid=gene_to_rowid,
            tgt_pert_tokens=tgt_pert_tokens,
        )

        self.generate = generate
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.iterations = iterations

        for key in [
            # 'cls_cosine_similarity',
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

    def delete_token(
        self,
        tgt_input_ids: torch.Tensor,
        token_to_perturb: torch.Tensor,
    ):
        # Create a mask for elements not equal to the target token
        mask = tgt_input_ids != token_to_perturb
        # add padding mask
        pad_mask = tgt_input_ids != self.pad_token_id
        mask_ = mask & pad_mask
        # Count the number of valid tokens in each sequence
        valid_counts = mask_.sum(dim=1)
        # Get indices for valid tokens
        valid_tokens = tgt_input_ids[mask_]

        # Initialize the result tensor filled with pad_token_id
        tgt_input_ids = torch.full_like(tgt_input_ids, self.pad_token_id)

        # Use advanced indexing to fill the valid tokens
        # into the tgt_input_ids tensor
        batch_indices = torch.arange(
            tgt_input_ids.size(0), device=tgt_input_ids.device
        ).repeat_interleave(valid_counts)
        position_indices = torch.cat(
            [torch.arange(c, device=tgt_input_ids.device) for c in valid_counts]
        )
        tgt_input_ids[batch_indices, position_indices] = valid_tokens
        return tgt_input_ids

    def forward(
        self,
        batch: Any,
        perturbation: bool = False,
    ):
        if perturbation:
            if self.perturbation_mode in ['mask', 'pad']:
                if self.gene_to_tgtid is not None:
                    if self.perturbation_mode == 'mask':
                        perturbation_token = self.gene_to_tgtid['<mask>']
                    elif self.perturbation_mode == 'pad':
                        perturbation_token = self.gene_to_tgtid['<pad>']
                    replace_token = torch.tensor(
                        perturbation_token, dtype=torch.long, device=self.device
                    )
                else:
                    raise ValueError(
                        'Please specify the mapping_dict path'
                        'to map the perturbation token'
                    )

        for i in self.pred_tps:
            tgt_input_id_ = batch[f'tgt_input_ids_t{i}'].clone()
            tgt_input_id_dict = {}
            if perturbation:
                if (
                    self.perturbation_sequence is not None
                    and 'tgt' in self.perturbation_sequence
                ):
                    mask = torch.isin(tgt_input_id_, self.tgt_pert_tokens)
                    if self.perturbation_mode in ['delete', 'overexpress']:
                        tgt_input_id_ = self.delete_token(
                            tgt_input_id_, self.tgt_pert_tokens
                        )
                        # add another if to concatenate overexpressed genes
                        if self.perturbation_mode == 'overexpress':
                            if self.tgt_pert_tokens is not None:
                                # exclude padding token to keep the same sequence length
                                tgt_input_id_ = tgt_input_id_[:, :-1]
                                tgt_input_id_ = torch.cat(
                                    (
                                        self.tgt_pert_tokens.expand(
                                            tgt_input_id_.shape[0], -1
                                        ),
                                        tgt_input_id_,
                                    ),
                                    dim=1,
                                )
                    else:
                        tgt_input_id_[mask] = replace_token

            cond_ids = concat_cond_tokens(
                batch=batch,
                time_step=i,
                condition_dict=self.condition_dict,
            )
            tgt_input_id_ = torch.cat((cond_ids, tgt_input_id_), dim=1)
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if perturbation:
            if (
                self.perturbation_sequence is not None
                and 'src' in self.perturbation_sequence
            ):
                perturbed_src = batch['src_input_ids'].clone()
                if self.perturbation_mode in ['delete', 'overexpress']:
                    perturbed_src = self.delete_token(
                        perturbed_src, self.src_pert_tokens
                    )
                    if self.perturbation_mode == 'overexpress':
                        # exclude padding token to keep the same sequence length
                        perturbed_src = perturbed_src[:, :-1]
                        src_pert_tokens = self.src_pert_tokens.expand(
                            perturbed_src.shape[0], -1
                        )
                        # concatenate overexpressed genes after cl[s
                        perturbed_src = torch.cat(
                            (
                                perturbed_src[:, 0:1],
                                src_pert_tokens,
                                perturbed_src[:, 1:],
                            ),
                            dim=1,
                        )
                else:
                    mask = torch.isin(perturbed_src, self.src_pert_tokens)
                    perturbed_src[mask] = replace_token

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

    def test_step(self, batch, *args, **kwargs):
        if self.validation_mode == 'inference':
            true_outputs, _, true_ids_dict = self.forward(batch, perturbation=False)
            (
                perturbed_outputs,
                _,
                perturbed_ids_dict,
            ) = self.forward(batch, perturbation=True)
        elif self.validation_mode == 'generate':
            # print(self.transformer)
            # self.transformer = self.quantize_model(self.transformer)
            (
                _,
                pert_src_input_ids,
                perturbed_ids_dict,
            ) = self.forward(batch, perturbation=True)
            decoder_kwargs = {
                'tgt_input_id_dict': perturbed_ids_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
            }

            true_outputs, true_ids_dict = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                genes_to_perturb=self.tgt_pert_tokens,
                **decoder_kwargs,
            )
            perturbed_outputs, generated_ids_dict = self.transformer.generate(
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
            cond_len = len(self.condition_dict)
            true_gene = true_outputs[t]['dec_embedding'][:, cond_len:, :]
            # true_logits = true_outputs[t]['dec_logits'][:, 1:, :]
            perturbed_gene = perturbed_outputs[t]['dec_embedding'][:, cond_len:, :]

            # perturbed_logits = perturbed_outputs[t]['dec_logits'][:, 1:, :]

            # true_probs = torch.softmax(true_logits, dim=-1)
            # perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
            # delta_probs = torch.abs(true_probs - perturbed_probs)
            # token_probs_change = delta_probs.sum(dim=-1)
            # delta_gene_probs, self.marker_genes = map_results_to_genes(
            #     token_probs_change,
            #     mapping_dict=self.gene_to_tgtid,
            #     token_ids=batch[f'tgt_input_ids_t{t}'],
            # )
            gene_cos_sim = cosine_similarity(
                true_gene,
                perturbed_gene,
                dim=-1,
            )

            gene_cos_sim, self.marker_genes = map_results_to_genes(
                gene_cos_sim,
                mapping_dict=self.gene_to_tgtid,
                token_ids=batch[f'tgt_input_ids_t{t}'],
            )
            true_mean_embs = mean_nonpadding_embs(
                embs=true_gene,  # exclude cls token
                input_ids=batch[f'tgt_input_ids_t{t}'],
                mapping_dict=self.gene_to_tgtid,
                condition_dict=self.condition_dict,
                perturbation_tokens=self.tgt_pert_tokens,
                dim=1,
            )
            perturbed_mean_cls = mean_nonpadding_embs(
                embs=perturbed_gene,  # exclude cls token
                input_ids=batch[f'tgt_input_ids_t{t}'],
                mapping_dict=self.gene_to_tgtid,
                condition_dict=self.condition_dict,
                perturbation_tokens=self.tgt_pert_tokens,
                perturbation_mode=self.perturbation_mode,
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

            # cls_cos_sim = cls_cos_sim.detach().cpu().to(torch.float16)
            mean_cos_sim = mean_cos_sim.detach().cpu().to(torch.float16)
            gene_cos_sim = gene_cos_sim.detach().cpu().to(torch.float16)
            # true_cls = true_cls.detach().cpu().to(torch.float16)
            # perturbed_cls = perturbed_cls.detach().cpu().to(torch.float16)

            # token_probs_change = token_probs_change.detach().cpu().to(torch.float16)
            # delta_probs = delta_probs.detach().cpu().to(torch.float16)
            # delta_gene_probs = delta_gene_probs.detach().cpu().to(torch.float16)
            # self.test_dict['cls_cosine_similarity'].append(cls_cos_sim)
            self.test_dict['mean_cosine_similarity'].append(mean_cos_sim)
            self.test_dict['gene_cosine_similarity'].append(gene_cos_sim)
            # self.test_dict['true_cls'].append(true_cls)
            # self.test_dict['perturbed_cls'].append(perturbed_cls)

            # if self.gene_module_list is not None:
            #     true_gm_embs = return_gene_embeddings(
            #         true_gene,
            #         self.gene_module_dict,
            #         batch[f'tgt_input_ids_t{t}'],
            #     )
            #     perturbed_gm_embs = return_gene_embeddings(
            #         perturbed_gene,
            #         self.gene_module_dict,
            #         batch[f'tgt_input_ids_t{t}'],
            #     )
            #     true_background_embs = return_gene_embeddings(
            #         true_gene,
            #         self.background_gene_dict,
            #         batch[f'tgt_input_ids_t{t}'],
            #     )
            #     perturbed_background_embs = return_gene_embeddings(
            #         perturbed_gene,
            #         self.background_gene_dict,
            #         batch[f'tgt_input_ids_t{t}'],
            #     )
            # # convert float32 to calculate wasserstein distance
            # true_gm_embs = true_gm_embs.detach().cpu().to(torch.float32)
            # perturbed_gm_embs = perturbed_gm_embs.detach().cpu().to(torch.float32)
            # true_background_embs = (
            #     true_background_embs.detach().cpu().to(torch.float32)
            # )
            # perturbed_background_embs = (
            #     perturbed_background_embs.detach().cpu().to(torch.float32)
            # )
            # self.test_dict['true_gm_embs'].append(true_gm_embs)
            # self.test_dict['perturbed_gm_embs'].append(perturbed_gm_embs)
            # self.test_dict['true_background_embs'].append(true_background_embs)
            # self.test_dict['perturbed_background_embs'].append(
            #     perturbed_background_embs
            # )
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
        # # compute emd of gm_embs based on condition
        # if self.gene_module_list is not None:
        #     true_gm_embs = torch.cat(self.test_dict['true_gm_embs'], dim=0)
        #     perturbed_gm_embs = torch.cat(self.test_dict['perturbed_gm_embs'], dim=0)
        #     true_background_embs = torch.cat(
        #         self.test_dict['true_background_embs'], dim=0
        #     )
        #     perturbed_background_embs = torch.cat(
        #         self.test_dict['perturbed_background_embs'], dim=0
        #     )
        #     # compute the wasserstein distance per condition
        #     gm_wd = {}
        #     background_wd = {}
        #     conditions = np.concatenate(self.test_dict['cell_type'])
        #     for condition in np.unique(conditions):
        #         true_gm_embs_cond = true_gm_embs[conditions == condition]
        #         perturbed_gm_embs_cond = perturbed_gm_embs[conditions == condition]
        #         true_background_embs_cond = true_background_embs[
        #             conditions == condition
        #         ]
        #         perturbed_background_embs_cond = perturbed_background_embs[
        #             conditions == condition
        #         ]
        #         # non-zero mean aggregation
        #         true_gm_embs_cond = self.compute_non_zero_mean(true_gm_embs_cond)
        #         perturbed_gm_embs_cond = self.compute_non_zero_mean(
        #             perturbed_gm_embs_cond
        #         )
        #         true_background_embs_cond = self.compute_non_zero_mean(
        #             true_background_embs_cond
        #         )
        #         perturbed_background_embs_cond = self.compute_non_zero_mean(
        #             perturbed_background_embs_cond
        #         )
        #         # compute the wasserstein distance
        #         gm_wd[condition] = wasserstein(
        #             true_gm_embs_cond,
        #             perturbed_gm_embs_cond,
        #         )
        #         background_wd[condition] = wasserstein(
        #             true_background_embs_cond,
        #             perturbed_background_embs_cond,
        #         )
        #         # store results as dataframes and merge them
        #         gm_wd_df = pd.DataFrame.from_dict(
        #             gm_wd, orient='index', columns=['gm_wd']
        #         )
        #         background_wd_df = pd.DataFrame.from_dict(
        #             background_wd, orient='index', columns=['background_wd']
        #         )
        #         wd_df = pd.concat([gm_wd_df, background_wd_df], axis=1)
        #         # plot the results
        #         wd_df.plot(kind='bar')
        #         wd_df.to_csv(f'{self.output_dir}/wasserstein_distance.csv')

        #         del (
        #             true_gm_embs_cond,
        #             perturbed_gm_embs_cond,
        #             true_background_embs_cond,
        #             perturbed_background_embs_cond,
        #         )

        obs_key = self.var_list if len(self.var_list) > 0 else []
        obs_key.extend(['cell_idx'])
        if self.genes_to_perturb is not None:
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
