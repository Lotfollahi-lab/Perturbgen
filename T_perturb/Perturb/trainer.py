import pickle
from typing import (
    Any,
    Dict,
    List,
    Literal,
)

import evaluate
import torch
import anndata as ad
import numpy as np


# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity
from torchmetrics import MeanSquaredError

from T_perturb.Model.trainer import CytoMeisterTrainer, CountDecoderTrainer
from T_perturb.Perturb.T_model import PerturberMasking, PerturberCountDecoder
from T_perturb.Modules.T_model import CountDecoder, CytoMeister
from T_perturb.src.utils import (
    compute_rouge_score,
    concat_cond_tokens,
    map_results_to_genes,
#    mask_duplicates_across_batches,
#    mask_duplicates_within_batches,
    mean_nonpadding_embs,
    return_perturbation_adata,
)


class PerturberTrainer(CountDecoderTrainer):
    def __init__(
        self,
        generate: bool = False,
        sequence_length: int = 2048,
        temperature: float = 2.0,
        context_mode: bool = True,
        iterations: int = 18,
        mapping_dict_path: str | None = None,
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
        context_tps: List[int] | None = None, 
        return_attn: bool = False,
        tgt_adata: ad.AnnData | None = None,
        var_list: List[str] | None = None,
        #ckpt_masking_path: str | None = None,
        # gene_module_list: List[str] | None = None,
        # num_of_background_genes: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(mapping_dict_path=mapping_dict_path, *args, **kwargs)


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
            context_tps=context_tps if 'context_tps' in kwargs else None,
            n_total_tps=kwargs['n_total_tps'],
            encoder=kwargs['encoder'],
            encoder_path=kwargs['encoder_path'],
            mask_scheduler=kwargs['mask_scheduler'],
            pos_encoding_mode=kwargs['pos_encoding_mode'],
            return_attn=return_attn,
            context_mode=context_mode,
            condition_dict=kwargs['condition_dict'],
            gene_to_rowid=gene_to_rowid,
            tgt_pert_tokens=tgt_pert_tokens,
        )
        self.return_attn = return_attn
        self.decoder = PerturberCountDecoder(
            pretrained_model=self.pretrained_model ,
            loss_mode=kwargs["loss_mode"],
            d_condc=kwargs["d_condc"],
            d_condt=kwargs["d_condt"],
            layer_norm=kwargs["layer_norm"],
            max_seq_length=kwargs["max_seq_length"],
            use_positional_encoding=kwargs["use_positional_encoding"],
            encoder=kwargs["encoder"],
            pos_encoding_mode=kwargs["pos_encoding_mode"],
            add_cell_time=kwargs["add_cell_time"],
            d_model=kwargs["d_model"],
            dropout=kwargs["dropout"],
            pred_tps=kwargs["pred_tps"],
            context_tps=context_tps,
            n_total_tps=kwargs["n_total_tps"],
            n_genes=kwargs["n_genes"],
        )

        
        self.mse = MeanSquaredError()
        # total_vocab_size = tgt_vocab_size
        self.pred_tps = kwargs["pred_tps"]
        if context_tps is None:
            self.total_tps = kwargs["pred_tps"]
        else:
            self.total_tps = context_tps + kwargs["pred_tps"]
        self.generate = generate
        self.adata = tgt_adata
        
        self.test_dict: Dict[str, List[Any]] = {
            'true_counts': [],
            'ctrl_counts': [],
            'pred_counts': [],
            'cls_embeddings': [],
            'cell_idx': [],
        }
        
        if var_list is not None:
            self.var_list = var_list
            for var in self.var_list:
                self.test_dict[var] = []
        else:
            self.var_list = []
        
        self.use_count_decoder = use_count_decoder

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

    def on_load_checkpoint(self, checkpoint):
        # Inspect state_dict entries
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove unexpected keys: "theta" and "onehot"
        keys_to_remove = ["theta", "onehot"]
        for key in keys_to_remove:
            if key in state_dict:
                print(f"Removing {key} from checkpoint")
                del state_dict[key]  # Remove unwanted keys

    def delete_token(
        self,
        input_ids: torch.Tensor,
        token_to_perturb: torch.Tensor,
    ):
        # Create a mask for elements not equal to the target token
        mask = input_ids != token_to_perturb
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
                input_ids = input_ids[:, :-1]
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
                    replace_token = self.gene_to_tgtid['<pad>']
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

        if self.validation_mode == 'inference':
            outputs = self.pretrained_model(
                src_input_id=perturbed_src,
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=True,
            )
            if self.use_count_decoder:
                count_output = self.decoder(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict
                )
        else:
            outputs = self.pretrained_model.forward(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=self.return_embeddings,
            )
            if self.use_count_decoder:
                count_output = self.decoder(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict
                )
        if self.use_count_decoder:
            return outputs, perturbed_src, tgt_input_id_dict, count_output
        else:
            return outputs, perturbed_src, tgt_input_id_dict

    def test_step(self, batch, *args, **kwargs):
        if self.validation_mode == 'inference':
            if self.use_count_decoder:
                true_outputs, _, true_ids_dict, true_counts = self.forward(batch, perturbation=False)
            else:
                true_outputs, _, true_ids_dict = self.forward(batch, perturbation=False)
                true_counts = None  # Set to None if not using count decoder
            if self.use_count_decoder:
                (
                    perturbed_outputs,
                    _,
                    perturbed_ids_dict,
                    pred_counts,
                ) = self.forward(batch, perturbation=True)
            else:
                (
                    perturbed_outputs,
                    _,
                    perturbed_ids_dict,
                ) = self.forward(batch, perturbation=True)
                pred_counts = None  # Set to None when not using count decoder
        elif self.validation_mode == 'generate':
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

            true_outputs, true_ids_dict = self.pretrained_model.generate(
                src_input_id=batch['src_input_ids'],
                genes_to_perturb=self.tgt_pert_tokens,
                **decoder_kwargs,
            )
            perturbed_outputs, generated_ids_dict = self.pretrained_model.generate(
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
            ) # for genes
            #cell_idx = np.array(batch[f'tgt_cell_idx_t{t}'])
            
            # Debugging: Check shapes before filtering
            #print(f"DEBUG: Initial cell_idx shape: {cell_idx.shape}")

            # if len(self.test_dict['cell_idx']) == 0:
            #     all_cell_idx = np.array([])
            # else:
            #     all_cell_idx = np.concatenate(self.test_dict['cell_idx'])
            # if self.tgt_pert_tokens is not None:
            #     # remove cells where perturbed gene is not present
            #     tgt_mask = torch.isin(
            #         batch[f'tgt_input_ids_t{t}'], self.tgt_pert_tokens
            #     )
            #     tgt_mask = tgt_mask.sum(dim=1).detach().cpu().numpy()
            # if self.src_pert_tokens is not None:
            #     src_mask = torch.isin(batch['src_input_ids'], self.src_pert_tokens)
            #     src_mask = src_mask.sum(dim=1).detach().cpu().numpy()
            # # combine tgt and src mask
            # if self.tgt_pert_tokens is not None and self.src_pert_tokens is not None:
            #     # use logical OR to combine tgt and src mask
            #     remove_cells_wo_pert = tgt_mask + src_mask > 0
            # elif self.tgt_pert_tokens is not None:
            #     remove_cells_wo_pert = tgt_mask > 0
            # elif self.src_pert_tokens is not None:
            #     remove_cells_wo_pert = src_mask > 0
            # else:
            #     remove_cells_wo_pert = np.ones_like(cell_idx, dtype=bool)
            # cell_idx = cell_idx[remove_cells_wo_pert]
            # print(f"DEBUG: Filtered cell_idx shape: {cell_idx.shape}")
            # (dupl_outside_batch, cell_idx_filter_) = mask_duplicates_across_batches(
            #     all_cell_idx, cell_idx
            # )
            # (dupl_within_batch, cell_idx_filter_) = mask_duplicates_within_batches(
            #     cell_idx_filter_
            # )
            
            mean_cos_sim = mean_cos_sim.detach().cpu().to(torch.float16)
            gene_cos_sim = gene_cos_sim.detach().cpu().to(torch.float16)
            true_mean_embs = true_mean_embs.detach().cpu().to(torch.float16)
            perturbed_mean_cls = perturbed_mean_cls.detach().cpu().to(torch.float16)
            # remove duplicates
            # mean_cos_sim = mean_cos_sim[remove_cells_wo_pert]
            # gene_cos_sim = gene_cos_sim[remove_cells_wo_pert]
            # true_mean_embs = true_mean_embs[remove_cells_wo_pert]
            # perturbed_mean_cls = perturbed_mean_cls[remove_cells_wo_pert]

            # mean_cos_sim = mean_cos_sim[dupl_outside_batch]
            # gene_cos_sim = gene_cos_sim[dupl_outside_batch]
            # true_mean_embs = true_mean_embs[dupl_outside_batch]
            # perturbed_mean_cls = perturbed_mean_cls[dupl_outside_batch]
            # mean_cos_sim = mean_cos_sim[dupl_within_batch]
            # gene_cos_sim = gene_cos_sim[dupl_within_batch]
            # true_mean_embs = true_mean_embs[dupl_within_batch]
            # perturbed_mean_cls = perturbed_mean_cls[dupl_within_batch]
            
            self.test_dict['mean_cosine_similarity'].append(mean_cos_sim)
            self.test_dict['gene_cosine_similarity'].append(gene_cos_sim)
            self.test_dict['true_cls'].append(true_mean_embs)
            self.test_dict['perturbed_cls'].append(perturbed_mean_cls)

            if self.use_count_decoder:
                key_true = f"count_output_t{t}"
                key_pred = f"count_output_t{t}"

                if key_true in true_counts and key_pred in pred_counts:
                    tensor_true = true_counts[key_true]["count_mean"]  # Extract specific tensor
                    tensor_pred = pred_counts[key_pred]["count_mean"]

                    self.test_dict['true_counts'].append(tensor_true.detach().cpu())
                    self.test_dict['pred_counts'].append(tensor_pred.detach().cpu())
                else:
                    print(f"WARNING: {key_true} or {key_pred} not found in true_counts/pred_counts!")


            #self.test_dict['cell_idx'].append(cell_idx_filter_)
            self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{t}'])

            if len(self.var_list) > 0:
                for var in self.var_list:
                    self.test_dict[var].append(batch[f'{var}_t{t}'])
                    # remove duplicates
                    # var_values = np.array(batch[f'{var}_t{t}'])
                    # var_values = var_values[remove_cells_wo_pert]
                    # var_values = var_values[dupl_outside_batch]
                    # var_values = var_values[dupl_within_batch]
                    # self.test_dict[var].append(var_values)

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

        obs_key = self.var_list if len(self.var_list) > 0 else []
        obs_key.extend(['cell_idx'])
        if self.genes_to_perturb is not None:
            if len(self.genes_to_perturb) > 1:
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
            use_count_decoder = self.use_count_decoder,
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




class OLDPerturberTrainer(CytoMeisterTrainer):
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
        pert_tps: List[int] | None = None,
        exclude_src: bool = False,
        tokenid_to_rowid_path: str | None = None,
        use_count_decoder: bool = False,
        d_condc: int = 128,
        d_condt: int = 128,
        layer_norm: bool = False,
        use_positional_encoding: bool = False,
        add_cell_time: bool = True,
        n_genes: int = 100,
        loss_mode: Literal['mse', 'zinb'] = 'mse',
        lr: float = 0.001,
        weight_decay: float = 0.001,
        #ckpt_count_path: str | None = None,
        #ckpt_masking_path: str | None = None,
        # gene_module_list: List[str] | None = None,
        # num_of_background_genes: int | None = None,
        *args,
        **kwargs,
    ):
        
        super().__init__(*args, **kwargs)

        # List of arguments meant for CountDecoder
        # count_decoder_args = [
        #     'loss_mode', 'd_condc', 'd_condt', 'layer_norm',
        #     'use_positional_encoding','add_cell_time', 'n_genes'
        # ]
        # # Extract CountDecoder-related arguments
        # count_decoder_kwargs = {key: kwargs.pop(key, None) for key in count_decoder_args}
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
            return_attn=self.return_attn,
            context_mode=kwargs['context_mode'],
            condition_dict=kwargs['condition_dict'],
            gene_to_rowid=gene_to_rowid,
            tgt_pert_tokens=tgt_pert_tokens,
        )
        self.use_count_decoder = use_count_decoder
        print('d_condc',d_condc)
        if use_count_decoder:
            self.decoder = CountDecoder(
                pretrained_model=self.transformer,
                encoder=kwargs['encoder'],
                d_model=kwargs['d_model'],
                pred_tps=kwargs['pred_tps'],
                context_tps=kwargs['context_tps'] if 'context_tps' in kwargs else None,
                n_total_tps=kwargs['n_total_tps'],
                d_condc = d_condc,
                d_condt = d_condt,
                layer_norm = layer_norm,
                use_positional_encoding = use_positional_encoding,
                add_cell_time = add_cell_time,
                n_genes = n_genes,
                loss_mode = loss_mode,
                lr = lr,
                weight_decay = weight_decay,
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

    def on_load_checkpoint(self, checkpoint):
        if 'decoder' in checkpoint:
            state_dict_ = modify_ckpt_state_dict(checkpoint, 'decoder.')
            self.load_state_dict(state_dict_, strict=False)

    def delete_token(
        self,
        input_ids: torch.Tensor,
        token_to_perturb: torch.Tensor,
    ):
        # Create a mask for elements not equal to the target token
        mask = input_ids != token_to_perturb
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
                input_ids = input_ids[:, :-1]
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
                    replace_token = self.gene_to_tgtid['<pad>']
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

            cond_ids = concat_cond_tokens(
                batch=batch,
                time_step=i,
                condition_dict=self.condition_dict,
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

        if self.validation_mode == 'inference':
            outputs = self.transformer(
                src_input_id=perturbed_src,
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=True,
            )
            if self.use_count_decoder:
                count_output = self.decoder(
                    src_input_id=perturbed_src,
                    tgt_input_id_dict=tgt_input_id_dict
                )
        else:
            outputs = self.transformer.forward(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=self.return_embeddings,
            )
        if self.use_count_decoder:
            return outputs, perturbed_src, tgt_input_id_dict, count_output
        else:
            return outputs, perturbed_src, tgt_input_id_dict

    def test_step(self, batch, *args, **kwargs):
        if self.validation_mode == 'inference':
            if self.use_count_decoder:
                true_outputs, _, true_ids_dict, true_count = self.forward(batch, perturbation=False)
            else:
                true_outputs, _, true_ids_dict = self.forward(batch, perturbation=False)
                true_count = None  # Set to None if not using count decoder
            if self.use_count_decoder:
                (
                    perturbed_outputs,
                    _,
                    perturbed_ids_dict,
                    pred_counts,
                ) = self.forward(batch, perturbation=True)
            else:
                (
                    perturbed_outputs,
                    _,
                    perturbed_ids_dict,
                ) = self.forward(batch, perturbation=True)
                pred_counts = None  # Set to None when not using count decoder
        elif self.validation_mode == 'generate':
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
            ) # for genes
            mean_cos_sim = mean_cos_sim.detach().cpu().to(torch.float16)
            gene_cos_sim = gene_cos_sim.detach().cpu().to(torch.float16)
            true_mean_embs = true_mean_embs.detach().cpu().to(torch.float16)
            perturbed_mean_cls = perturbed_mean_cls.detach().cpu().to(torch.float16)
            self.test_dict['mean_cosine_similarity'].append(mean_cos_sim)
            self.test_dict['gene_cosine_similarity'].append(gene_cos_sim)
            self.test_dict['true_cls'].append(true_mean_embs)
            self.test_dict['perturbed_cls'].append(perturbed_mean_cls)

            if self.decoder:
                self.test_dict['true_count_output'].append(true_count_output.detach().cpu())
                self.test_dict['pred_counts'].append(pred_counts.detach().cpu())
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
