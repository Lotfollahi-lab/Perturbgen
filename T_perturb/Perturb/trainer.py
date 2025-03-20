import pickle
from typing import (
    Any,
    List,
    Literal,
)

import evaluate
import numpy as np
import torch

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CountDecoderTrainer
from T_perturb.Perturb.T_model import PerturberCountDecoder, PerturberMasking
from T_perturb.src.utils import (
    compute_rouge_score,
    concat_cond_tokens,
    map_results_to_genes,
    return_perturbation_adata,
)


class PerturberTrainer(CountDecoderTrainer):
    def __init__(
        self,
        genes_to_perturb: List[str] | None = None,
        src_tokens_to_perturb: List[int] | None = None,
        tgt_tokens_to_perturb: List[int] | None = None,
        validation_mode: Literal['inference', 'generate'] | None = None,
        perturbation_mode: Literal['mask', 'pad', 'delete', 'overexpress']
        | None = None,
        perturbation_sequence: Literal['src', 'tgt'] | None = None,
        pert_tps: List[int] | None = None,
        exclude_src: bool = False,
        tokenid_to_rowid_path: str | None = None,
        use_count_decoder: bool = False,
        pad_condition: bool = False,
        batch_size: int = 16,
        # gene_module_list: List[str] | None = None,
        # num_of_background_genes: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.validation_mode = validation_mode
        self.pad_condition = pad_condition
        if validation_mode is not None:
            if perturbation_sequence is None:
                raise ValueError(
                    'Please specify the perturbation_sequence: "src" or "tgt"'
                )
            if perturbation_mode is None:
                raise ValueError('Please specify the perturbation_token')

            self.perturbation_sequence = perturbation_sequence
            self.perturbation_mode = perturbation_mode

        # dictionary to map gene names to row ids in tgt
        if kwargs['mapping_dict_path'] is not None:
            with open(
                kwargs['mapping_dict_path'],
                'rb',
            ) as f:
                rowid_to_gene = pickle.load(f)
            gene_to_rowid = {v: k for k, v in rowid_to_gene.items()}
            self.gene_to_tgtid = gene_to_rowid

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
            else:
                self.src_pert_tokens = None
            if 'tgt' in perturbation_sequence:
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
            self.register_buffer('tgt_pert_tokens', tgt_pert_tokens, persistent=False)
            self.genes_to_perturb = genes_to_perturb
        else:
            raise ValueError('Please specify the perturbation_sequence: "src" or "tgt"')
        self.pert_tps: list[int] | None = None
        if pert_tps is not None:
            # check if pert_tps are in pred_tps
            if not all(tp in self.pred_tps for tp in pert_tps):
                raise ValueError(
                    f'Time steps in pert_tps: {pert_tps} '
                    f'must be in pred_tps: {self.pred_tps}'
                )
            else:
                self.pert_tps = pert_tps

        self.exclude_src = exclude_src
        print(
            f'Start perturbation ...\n'
            f'- Validation mode: {self.validation_mode}\n'
            f'- Perturbation sequence: {self.perturbation_sequence}\n'
            f'- Perturbing genes: {genes_to_perturb}\n'
            f'- Perturbation mode: {perturbation_mode}\n'
            f'- Perturbation tps: {pert_tps}\n'
        )
        self.pretrained_model = PerturberMasking(
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
            context_mode=kwargs['context_mode'],
            condition_dict=kwargs['condition_dict'],
            gene_to_rowid=gene_to_rowid,
        )
        if use_count_decoder:
            self.decoder = PerturberCountDecoder(
                pretrained_model=self.pretrained_model,
                loss_mode=kwargs['loss_mode'],
                d_condc=kwargs['d_condc'] if 'd_condc' in kwargs else None,
                d_condt=kwargs['d_condt'] if 'd_condt' in kwargs else None,
                layer_norm=kwargs['layer_norm'] if 'layer_norm' in kwargs else False,
                max_seq_length=kwargs['max_seq_length'],
                use_positional_encoding=(
                    kwargs['use_positional_encoding']
                    if 'use_positional_encoding' in kwargs
                    else False
                ),
                encoder=kwargs['encoder'],
                pos_encoding_mode=kwargs['pos_encoding_mode'],
                add_cell_time=kwargs['add_cell_time']
                if 'add_cell_time' in kwargs
                else None,
                d_model=kwargs['d_model'],
                dropout=kwargs['dropout'],
                pred_tps=kwargs['pred_tps'],
                context_tps=kwargs['context_tps'] if 'context_tps' in kwargs else None,
                n_total_tps=kwargs['n_total_tps'],
                n_genes=kwargs['n_genes'],
            )

        for key in [
            # 'cls_cosine_similarity',
            'mean_cosine_similarity',
            'mean_cosine_similarity_l1',
            'mean_cosine_similarity_lmid',
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
            'true_counts',
            'pred_counts',
            'pert_counts',
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
        self.use_count_decoder = use_count_decoder
        self.pad_token_id = self.gene_to_tgtid['<pad>']

    def on_load_checkpoint(self, checkpoint):
        # Inspect state_dict entries
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove unexpected keys: "onehot"
        keys_to_remove = ['onehot']
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]  # Remove unwanted keys

    def delete_token(
        self,
        input_ids: torch.Tensor,
        token_to_perturb: torch.Tensor,
    ):
        # Create a mask for elements not equal to the target token
        mask = ~torch.isin(input_ids, token_to_perturb)
        # add padding mask
        pad_mask = input_ids != self.pad_token_id
        mask_ = mask & pad_mask
        # Count the number of valid tokens in each sequence
        valid_counts = mask_.sum(dim=1)
        # Get indices for valid tokens
        valid_tokens = input_ids[mask_]
        # Initialize the result tensor filled with pad_token_id
        input_ids = torch.full_like(input_ids, self.pad_token_id)
        # Use advanced indexing to fill the valid tokens
        # into the input_ids tensor
        batch_indices = torch.arange(
            input_ids.size(0), device=input_ids.device
        ).repeat_interleave(valid_counts)
        position_indices = torch.cat(
            [torch.arange(c, device=input_ids.device) for c in valid_counts]
        )
        input_ids[batch_indices, position_indices] = valid_tokens
        return input_ids

    def perturb_sequence(
        self,
        input_ids: torch.Tensor,
        token_to_perturb: torch.Tensor,
        replace_token: torch.Tensor,
        perturbation_mode: str,
        perturbation_sequence: str,
    ):
        if (perturbation_mode == 'mask') or (perturbation_mode == 'pad'):
            mask = torch.isin(input_ids, token_to_perturb)
            input_ids[mask] = replace_token
        elif (perturbation_mode == 'delete') or (perturbation_mode == 'overexpress'):
            input_ids = self.delete_token(input_ids, token_to_perturb)
            if perturbation_mode == 'overexpress':
                # exclude padding token to keep the same sequence length
                input_ids = input_ids[:, : -len(token_to_perturb)]
                token_to_perturb = token_to_perturb.expand(input_ids.shape[0], -1)
                if perturbation_sequence == 'tgt':
                    input_ids = torch.cat(
                        (
                            token_to_perturb,
                            input_ids,
                        ),
                        dim=1,
                    )
                elif perturbation_sequence == 'src':
                    input_ids = torch.cat(
                        (
                            input_ids[:, 0:1],
                            token_to_perturb,
                            input_ids[:, 1:],
                        ),
                        dim=1,
                    )
        return input_ids

    def forward(
        self,
        batch: Any,
        perturbation: bool = False,
    ):
        if perturbation:
            if self.gene_to_tgtid is not None:
                replace_token: torch.Tensor | None = None
                if self.perturbation_mode == 'mask':
                    replace_token = self.gene_to_tgtid['<mask>']
                    replace_token = torch.tensor(
                        replace_token, dtype=torch.long, device=self.device
                    )
                elif self.perturbation_mode == 'pad':
                    replace_token = self.pad_token_id
                    replace_token = torch.tensor(
                        replace_token, dtype=torch.long, device=self.device
                    )

            else:
                raise ValueError(
                    'Please specify the mapping_dict path'
                    'to map the perturbation token'
                )
        tgt_input_id_dict = {}
        for i in self.pred_tps:
            tgt_input_id_ = batch[f'tgt_input_ids_t{i}'].clone()
            if perturbation:
                if (
                    self.perturbation_sequence is not None
                    and 'tgt' in self.perturbation_sequence
                ):
                    # perturb only specific time steps
                    if self.pert_tps is not None:
                        if i in self.pert_tps:
                            tgt_input_id_ = self.perturb_sequence(
                                tgt_input_id_,
                                self.tgt_pert_tokens,
                                replace_token,
                                self.perturbation_mode,
                                'tgt',
                            )
                    # perturb all time steps
                    else:
                        tgt_input_id_ = self.perturb_sequence(
                            tgt_input_id_,
                            self.tgt_pert_tokens,
                            replace_token,
                            self.perturbation_mode,
                            'tgt',
                        )
            if self.condition_dict is not None:
                cond_ids = concat_cond_tokens(
                    batch=batch,
                    time_step=i,
                    condition_dict=self.condition_dict,
                    pad_condition=self.pad_condition,
                )
                tgt_input_id_ = torch.cat((cond_ids, tgt_input_id_), dim=1)
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if self.exclude_src:
            # pad src_input_ids to ignore src
            perturbed_src = torch.zeros_like(batch['src_input_ids'])
        else:
            if perturbation:
                if (
                    self.perturbation_sequence is not None
                    and 'src' in self.perturbation_sequence
                ):
                    perturbed_src = batch['src_input_ids'].clone()
                    perturbed_src = self.perturb_sequence(
                        perturbed_src,
                        self.src_pert_tokens,
                        replace_token,
                        self.perturbation_mode,
                        'src',
                    )
                else:
                    perturbed_src = batch['src_input_ids']
            else:
                perturbed_src = batch['src_input_ids']
        tgt_pert_tokens_ = self.tgt_pert_tokens if perturbation else None
        if self.validation_mode == 'inference':
            if self.use_count_decoder is False:
                # true counts do not need to be computed
                outputs = self.pretrained_model(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict,
                    not_masked=True,
                    tgt_pert_tokens=tgt_pert_tokens_,
                )
                count_output = None
            else:
                outputs, count_outputs = self.decoder(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict,
                    tgt_pert_tokens=tgt_pert_tokens_,
                )
                _, count_output = self.compute_count_loss(
                    count_outputs, batch, n_samples=self.n_samples
                )

        else:
            if self.use_count_decoder is False:
                outputs = self.pretrained_model.forward(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict,
                    not_masked=self.return_embeddings,
                    tgt_pert_tokens=tgt_pert_tokens_,
                )
                count_output = None

            else:
                outputs, count_outputs = self.decoder(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict,
                    tgt_pert_tokens=tgt_pert_tokens_,
                )
                _, count_output = self.compute_count_loss(
                    count_outputs, batch, n_samples=self.n_samples
                )
        return (outputs, perturbed_src, tgt_input_id_dict, count_output)

    def apply_mask(self, x, mask):
        # If x is a PyTorch tensor:
        if isinstance(x, torch.Tensor):
            # Ensure mask is a tensor on the correct device and boolean:
            if not isinstance(mask, torch.Tensor):
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=x.device)
            else:
                mask_tensor = mask.to(x.device)
                if mask_tensor.dtype != torch.bool:
                    mask_tensor = mask_tensor.bool()
            return x[mask_tensor]
        # If x is a NumPy array:
        elif isinstance(x, np.ndarray):
            # Convert mask to a numpy boolean array:
            mask_array = np.asarray(mask, dtype=bool)
            return x[mask_array]
        # If x is a list:
        elif isinstance(x, list):
            # Convert mask to a list of booleans:
            return [x[i] for i, m in enumerate(mask) if m]
        else:
            raise TypeError(
                f'Unsupported type for x: {type(x)}.'
                'x must be a PyTorch tensor or NumPy array.'
            )

    def test_step(self, batch, *args, **kwargs):
        if self.tgt_pert_tokens is not None:
            pert_tps_ = self.pert_tps
            # remove cells where perturbed gene is not present

            tgt_mask_list = []
            for t in pert_tps_:
                tgt_mask = torch.isin(
                    batch[f'tgt_input_ids_t{t}'], self.tgt_pert_tokens
                )
                tgt_mask = tgt_mask.sum(dim=1).detach().cpu().numpy()
                tgt_mask_list.append(tgt_mask)
            tgt_mask = np.sum(tgt_mask_list, axis=0)

        if self.src_pert_tokens is not None:
            src_mask = torch.isin(batch['src_input_ids'], self.src_pert_tokens)
            src_mask = src_mask.sum(dim=1).detach().cpu().numpy()
        # combine tgt and src mask
        if self.tgt_pert_tokens is not None and self.src_pert_tokens is not None:
            # use logical OR to combine tgt and src mask
            remove_cells_wo_pert = tgt_mask + src_mask > 0
        elif self.tgt_pert_tokens is not None:
            remove_cells_wo_pert = tgt_mask > 0
        elif self.src_pert_tokens is not None:
            remove_cells_wo_pert = src_mask > 0
        else:
            remove_cells_wo_pert = np.ones(batch['src_input_ids'].shape[0], dtype=bool)
        # if all cells are removed where boolean mask is all False
        if np.sum(remove_cells_wo_pert) > 0:
            # remove cells without perturbation
            # filtered_batch = batch
            filtered_batch = {
                k: self.apply_mask(v, remove_cells_wo_pert)
                for k, v in batch.items()
                if v is not None
            }

            if self.validation_mode == 'inference':
                (true_outputs, _, true_ids_dict, pred_counts) = self.forward(
                    filtered_batch, perturbation=False
                )
                (
                    perturbed_outputs,
                    _,
                    perturbed_ids_dict,
                    pert_counts,
                ) = self.forward(filtered_batch, perturbation=True)
            elif self.validation_mode == 'generate':
                (
                    _,
                    pert_src_input_ids,
                    perturbed_ids_dict,
                    pert_counts,
                ) = self.forward(filtered_batch, perturbation=True)
                decoder_kwargs = {
                    'tgt_input_id_dict': perturbed_ids_dict,
                    'mask_scheduler': self.mask_scheduler,
                    'can_remask_prev_masked': False,
                    'topk_filter_thres': 0.9,
                    'temperature': self.temperature,
                    'iterations': self.iterations,
                }

                true_outputs, true_ids_dict = self.pretrained_model.generate(
                    src_input_id=filtered_batch['src_input_ids'],
                    genes_to_perturb=self.tgt_pert_tokens,
                    **decoder_kwargs,
                )
                perturbed_outputs, generated_ids_dict = self.pretrained_model.generate(
                    src_input_id=pert_src_input_ids,
                    **decoder_kwargs,
                )
                for t in self.pred_tps:
                    true_ids = true_ids_dict[t].detach().cpu().numpy()
                    # ground truth
                    input_ids = (
                        filtered_batch[f'tgt_input_ids_t{t}'].detach().cpu().numpy()
                    )
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
                cond_len = len(self.condition_dict)
                true_gene = true_outputs[t]['dec_embedding'][:, cond_len:, :]
                true_mean_embs = true_outputs[t]['mean_embedding']
                true_mean_embs_l1 = true_outputs[t]['mean_embedding_l1']
                true_mean_embs_lmid = true_outputs[t]['mean_embedding_lmid']
                perturbed_gene = perturbed_outputs[t]['dec_embedding'][:, cond_len:, :]
                perturbed_mean_embs = perturbed_outputs[t]['mean_embedding']
                perturbed_mean_embs_l1 = perturbed_outputs[t]['mean_embedding_l1']
                perturbed_mean_embs_lmid = perturbed_outputs[t]['mean_embedding_lmid']

                true_ids = true_ids_dict[f'tgt_input_ids_t{t}'][:, cond_len:]
                perturbed_ids = perturbed_ids_dict[f'tgt_input_ids_t{t}'][:, cond_len:]
                if (self.perturbation_mode == 'delete') or (
                    self.perturbation_mode == 'overexpress'
                ):
                    # TODO: see ChatGPT and complete the code
                    # create a mask for perturbed gene
                    if self.perturbation_mode == 'overexpress':
                        # remove overexpressed genes from true_gene and perturbed_gene
                        perturbed_ids[
                            :, : len(self.tgt_pert_tokens)
                        ] = self.pad_token_id

                    match_mask = true_ids.unsqueeze(1) == perturbed_ids.unsqueeze(
                        2
                    )  # (batch_size, seq_len, seq_len)
                    true_indices = match_mask.float().argmax(dim=2)  # shape: [B, T]
                    b, seq, emb_dim = true_gene.shape
                    true_ids = torch.gather(true_ids, 1, true_indices)
                    true_gene = torch.gather(
                        true_gene, 1, true_indices.unsqueeze(2).expand(b, seq, emb_dim)
                    )
                    if self.perturbation_mode == 'overexpress':
                        # remove overexpressed genes from true_gene and perturbed_gene
                        true_gene = true_gene[:, len(self.tgt_pert_tokens) :, :]
                        true_ids = true_ids[:, len(self.tgt_pert_tokens) :]
                        perturbed_gene = perturbed_gene[
                            :, len(self.tgt_pert_tokens) :, :
                        ]
                        perturbed_ids = perturbed_ids[:, len(self.tgt_pert_tokens) :]

                    # check if true_gene_ids is equal to perturbed_gene_ids
                    torch.allclose(true_ids, perturbed_ids)

                gene_cos_sim = cosine_similarity(
                    true_gene,
                    perturbed_gene,
                    dim=-1,
                )
                gene_cos_sim, self.marker_genes = map_results_to_genes(
                    gene_cos_sim,
                    mapping_dict=self.gene_to_tgtid,
                    token_ids=perturbed_ids,  # pass perturbed ids
                )
                mean_cos_sim = cosine_similarity(
                    perturbed_mean_embs,
                    true_mean_embs,
                )
                mean_cos_sim_l1 = cosine_similarity(
                    perturbed_mean_embs_l1,
                    true_mean_embs_l1,
                )
                mean_cos_sim_lmid = cosine_similarity(
                    perturbed_mean_embs_lmid,
                    true_mean_embs_lmid,
                )

                cell_idx = np.array(filtered_batch[f'tgt_cell_idx_t{t}'])

                mean_cos_sim = mean_cos_sim.detach().cpu().to(torch.float16)
                mean_cos_sim_l1 = mean_cos_sim_l1.detach().cpu().to(torch.float16)
                mean_cos_sim_lmid = mean_cos_sim_lmid.detach().cpu().to(torch.float16)
                gene_cos_sim = gene_cos_sim.detach().cpu().to(torch.float16)
                # true_cls = true_cls.detach().cpu().to(torch.float16)
                # perturbed_cls = perturbed_cls.detach().cpu().to(torch.float16)
                true_mean_embs = true_mean_embs.detach().cpu().to(torch.float16)
                perturbed_mean_embs = (
                    perturbed_mean_embs.detach().cpu().to(torch.float16)
                )

                # mean_cos_sim = mean_cos_sim[dupl_outside_batch]
                # mean_cos_sim_l1 = mean_cos_sim_l1[dupl_outside_batch]
                # mean_cos_sim_lmid = mean_cos_sim_lmid[dupl_outside_batch]
                # gene_cos_sim = gene_cos_sim[dupl_outside_batch]
                # true_mean_embs = true_mean_embs[dupl_outside_batch]
                # perturbed_mean_embs = perturbed_mean_embs[dupl_outside_batch]
                # mean_cos_sim = mean_cos_sim[dupl_within_batch]
                # mean_cos_sim_l1 = mean_cos_sim_l1[dupl_within_batch]
                # mean_cos_sim_lmid = mean_cos_sim_lmid[dupl_within_batch]
                # gene_cos_sim = gene_cos_sim[dupl_within_batch]
                # true_mean_embs = true_mean_embs[dupl_within_batch]
                # perturbed_mean_embs = perturbed_mean_embs[dupl_within_batch]
                self.test_dict['mean_cosine_similarity'].append(mean_cos_sim)
                self.test_dict['mean_cosine_similarity_l1'].append(mean_cos_sim_l1)
                self.test_dict['mean_cosine_similarity_lmid'].append(mean_cos_sim_lmid)
                self.test_dict['gene_cosine_similarity'].append(gene_cos_sim)
                self.test_dict['true_cls'].append(true_mean_embs)
                self.test_dict['perturbed_cls'].append(perturbed_mean_embs)
                if self.use_count_decoder:
                    if t in pert_counts:
                        true_counts_ = filtered_batch[f'tgt_counts_t{t}'].detach().cpu()
                        pred_counts_ = pred_counts[t].detach().cpu()
                        pert_counts_ = pert_counts[t].detach().cpu()
                        # true_counts_ = true_counts_[dupl_outside_batch]
                        # pred_counts_ = pred_counts_[dupl_outside_batch]
                        # pert_counts_ = pert_counts_[dupl_outside_batch]
                        # true_counts_ = true_counts_[dupl_within_batch]
                        # pred_counts_ = pred_counts_[dupl_within_batch]
                        # pert_counts_ = pert_counts_[dupl_within_batch]
                        self.test_dict['true_counts'].append(true_counts_)
                        self.test_dict['pred_counts'].append(pred_counts_)
                        self.test_dict['pert_counts'].append(pert_counts_)
                    else:
                        Warning(f'Counts are not available for time step: {t}')
                self.test_dict['cell_idx'].append(cell_idx)
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        # remove duplicates
                        var_values = np.array(filtered_batch[f'{var}_t{t}'])
                        # var_values = var_values[dupl_outside_batch]
                        # var_values = var_values[dupl_within_batch]
                        self.test_dict[var].append(var_values)
        else:
            pass

    def on_test_epoch_end(self):
        obs_key = self.var_list if len(self.var_list) > 0 else []
        obs_key.extend(['cell_idx'])
        if self.genes_to_perturb is not None:
            if len(self.genes_to_perturb) > 1:
                # concatenate maximum 5 genes if more than 5 genes are perturbed
                if len(self.genes_to_perturb) > 5:
                    genes_to_perturb = '_'.join(self.genes_to_perturb[:5])
                else:
                    genes_to_perturb = '_'.join(self.genes_to_perturb)
            else:
                genes_to_perturb = self.genes_to_perturb[0]
        else:
            genes_to_perturb = None
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
