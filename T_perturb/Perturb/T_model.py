from typing import Dict, List

import torch
from einops import rearrange
from torch import nn

from T_perturb.Modules.T_model import CytoMeister
from T_perturb.src.utils import (
    gumbel_sample,
    mean_nonpadding_embs,
    noise_schedule,
    top_k,
)


class PerturberMasking(CytoMeister):
    def __init__(
        self,
        tgt_pert_tokens,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_scheduler = kwargs['mask_scheduler']
        self.register_buffer('tgt_pert_tokens', tgt_pert_tokens, persistent=False)

    def generate(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id_dict: dict,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        # time_steps=[1, 2, 3],
        temperature: float = 2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        iterations: int = 18,  # optimal of iterations in MaskGIT
        cond_length: int = 0,
        mask_scheduler: str = 'cosine',
        sequence_length: int | None = None,
        genes_to_perturb: List | None = None,
        prompt_length: int = 1,
        **kwargs,
    ):
        '''
        Description:
        ------------
        Generate sequences for the target tokens using guided generation.

        Parameters:
        -----------
        src_input_id: `torch.Tensor`
            Source token input.
        tgt_input_id_dict: `dict`
            Dictionary of target token inputs from different time steps.
        max_len: `int`
            Maximum length of the generated sequence.
        can_remask_prev_masked: `bool`
            Whether to remask previously masked tokens.
        topk_filter_thres: `float`
            Top-k filter threshold based on the logits.
        temperature: `float`
            Temperature to increase or decrease the randomness of the predictions.
        iterations: `int`
            Number of iterations until all tokens are predicted.
        mask_scheduler: `str`
            Mask scheduler function.
            Options: ['uniform', 'pow', 'cosine', 'log', 'exp']
        genes_to_perturb: `List`
            List of genes to perturb.
        prompt_length: `int`
            Length of the prompt to be used for generation.

        Returns:
        --------
        count_outputs: `dict`
            Output dictionary containing the following keys:
            - 'count_output_t{t}': Count prediction for time step t.
            - 'cls_embedding_t{t}': CLS token embeddings for time step t.
        '''
        generate_id_dict: Dict[str, torch.Tensor] = {}
        all_outputs: Dict[int, torch.Tensor] = {}
        if self.context_tps is not None:
            all_modelling_tps = self.pred_tps + self.context_tps
            all_modelling_tps = sorted(list(set(all_modelling_tps)))
        else:
            all_modelling_tps = sorted(self.pred_tps)
        tgt_pad_dict = self.call_padding(tgt_input_id_dict, all_modelling_tps)
        # filter tgt_input_id_dict to include only all_modelling_tps
        tgt_input_id_dict = {
            k: v
            for k, v in tgt_input_id_dict.items()
            if int(k.split('_t')[-1]) in all_modelling_tps
        }
        for time_step in self.pred_tps:
            tgt_input_id_dict_ = {k: v.clone() for k, v in tgt_input_id_dict.items()}
            tgt_input_id_key = f'tgt_input_ids_t{time_step}'
            tgt_input_id = tgt_input_id_dict[tgt_input_id_key]
            # create ids and scores matrix for each batch
            ids = torch.full_like(tgt_input_id, self.mask_token, dtype=torch.long)
            # add cls token to the ids
            ids[:, :cond_length] = tgt_input_id[:, :cond_length]
            # pad ids based on tgt_pad_dict
            ids[tgt_pad_dict[f'tgt_pad_t{time_step}']] = self.pad_token
            tgt_input_id_dict_[tgt_input_id_key] = ids
            scores = torch.zeros_like(tgt_input_id, dtype=torch.float)
            # use a two-step process to decode the genes
            outputs, generated_ids = self.generate_sequence(
                generate_id_dict=tgt_input_id_dict_,
                generate_pad_dict=tgt_pad_dict,
                src_input_id=src_input_id,
                demask_fn=self,
                mask_scheduler=mask_scheduler,
                can_remask_prev_masked=can_remask_prev_masked,
                topk_filter_thres=topk_filter_thres,
                starting_temperature=temperature,
                iterations=iterations,
                scores=scores,
                tgt_time_step=time_step,
                prompt_length=prompt_length,
                genes_to_perturb=genes_to_perturb,
            )
            generate_id_dict[f'tgt_input_ids_t{time_step}'] = generated_ids
            all_outputs[time_step] = outputs
        return all_outputs, generate_id_dict

    def generate_sequence(
        self,
        generate_id_dict: dict,
        generate_pad_dict: dict,
        src_input_id: torch.Tensor,
        demask_fn: nn.Module,
        mask_scheduler: str,
        scores: torch.Tensor,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        starting_temperature: float = 2.0,
        iterations: int = 18,
        tgt_time_step: int = 1,
        prompt_length: int = 1,
        genes_to_perturb: List | None = None,
    ):
        '''
        Description:
        ------------
        Generate sequences for the target tokens
        adopted from MaskGIT using the pretrained model.
        Parameters:
        -----------
        generate_id_dict: `dict`
            Dictionary of target token inputs for generation.
        generate_pad_dict: `dict`
            Dictionary of target padding masks for generation.
        src_input_id: `torch.Tensor`
            Source token input.
        demask_fn: `nn.Module`
            Pretrained model for demasking.
        mask_scheduler: `str`
            Mask scheduler function.
            Options: ['uniform', 'pow', 'cosine', 'log', 'exp']
        scores: `torch.Tensor`
            Probability scores for the tokens.
        can_remask_prev_masked: `bool`
            Whether to remask previously masked tokens.
        topk_filter_thres: `float`
            Top-k filter threshold based on the logits.
        starting_temperature: `float`
            Temperature to increase or decrease the randomness of the predictions.
        iterations: `int`
            Number of iterations until all tokens are predicted.
        tgt_time_step: `int`
            Target time step to generate sequence for.
        Returns:
        --------
        `tuple` containing:
            outputs: `dict`
                Output dictionary containing the following keys:
                - 'dec_logits': Decoder logits.
                - 'labels': True labels for masked tokens.
                - 'selected_time_step': Selected time step.
                - 'dec_embedding': Decoder embeddings.
                - 'mean_embedding': Mean embeddings for non-padding tokens.
            tmp_ids: `torch.Tensor`
                Generated target token inputs.
        '''
        max_neg_value = -torch.finfo().max
        scores[:, :prompt_length] = max_neg_value
        tmp_ids = generate_id_dict[f'tgt_input_ids_t{tgt_time_step}'].clone()
        batch_size, seq_len = tmp_ids.shape
        # find total_tokens by find the numbers of 1s in the mask
        total_tokens = torch.sum(tmp_ids == 1, dim=1)
        ids_to_keep = torch.zeros_like(tmp_ids, dtype=torch.long)
        iteration_ratios = torch.linspace(0, 1, iterations)
        all_steps = reversed(range(iterations))
        for iteration, steps_until_x0 in zip(
            iteration_ratios,
            all_steps,
        ):
            # mask scheduler function, gamma
            rand_mask_prob = noise_schedule(
                ratio=iteration,
                total_tokens=total_tokens,
                method=mask_scheduler,
            )
            # set score to -inf for padding tokens
            scores.masked_fill_(
                generate_pad_dict[f'tgt_pad_t{tgt_time_step}'], max_neg_value
            )
            unmasked = (scores != max_neg_value).sum(dim=1)
            num_tokens_to_mask = (unmasked.float() * rand_mask_prob).long()
            mask = torch.zeros_like(scores, dtype=torch.bool)
            _, indices_to_mask = torch.topk(scores, num_tokens_to_mask.max(), dim=-1)
            # Mask the top `num_tokens_to_mask` positions for each sample
            for i in range(batch_size):
                mask[i, indices_to_mask[i, : num_tokens_to_mask[i]]] = True
            tmp_ids.masked_fill_(mask, self.mask_token)
            # keep indices which are not masked except for the CLS token
            ids_to_keep = torch.where(
                mask,
                torch.tensor(0, dtype=tmp_ids.dtype, device=tmp_ids.device),
                tmp_ids,
            )
            generate_id_dict[f'tgt_input_ids_t{tgt_time_step}'] = tmp_ids
            outputs = demask_fn.forward(
                src_input_id=src_input_id,  # target
                generate_id_dict=generate_id_dict,
                generate_pad_dict=generate_pad_dict,
                not_masked=False,
                tgt_time_step=tgt_time_step,
            )
            logits = outputs[tgt_time_step]['dec_logits'][:, prompt_length:, :]

            if genes_to_perturb is not None:
                # exclude genes_to_perturb from the option to be selected
                logits[:, :, genes_to_perturb] = max_neg_value
            # exclude cls token
            tmp_ids_ = tmp_ids[:, prompt_length:].clone()
            scores_ = scores[:, prompt_length:].clone()
            ids_to_keep_ = ids_to_keep[:, prompt_length:].clone()
            # Create a mask of already predicted tokens
            indices = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - prompt_length, -1)
            logits.scatter_(2, indices, max_neg_value)

            filtered_logits = top_k(logits.clone(), topk_filter_thres)
            temperature = starting_temperature * (
                steps_until_x0 / iteration
            )  # temperature is annealed
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = tmp_ids_ == self.mask_token
            tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)
            probs_without_temperature = logits.softmax(dim=-1)

            scores_ = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores_ = rearrange(scores_, '... 1 -> ...')

            if not can_remask_prev_masked:
                scores_.masked_fill_(~is_mask, max_neg_value)
            # add cls token to the ids and update scores and ids
            scores[:, prompt_length:] = scores_
            tmp_ids[:, prompt_length:] = tmp_ids_
        return outputs[tgt_time_step], tmp_ids

    def call_decoder(
        self,
        enc_output,
        src_attention_mask,
        dec_embedding,
        tgt_pad,
        labels=None,
        tgt_input_id=None,
    ):
        self_attn_list = []
        cross_attn_list = []
        for dec_layer in self.decoder_block:
            # see if concatenation of cls embedding
            dec_embedding, self_attn_weights, cross_attn_weights = dec_layer(
                x=dec_embedding,
                src_mask=src_attention_mask,
                tgt_mask=tgt_pad,
                enc_output=enc_output,
            )
            if self_attn_weights is not None:
                self_attn_list.append(self_attn_weights)
            if cross_attn_weights is not None:
                cross_attn_list.append(cross_attn_weights)
        # also convert to float 16 for memory efficiency
        if len(self_attn_list) > 0:
            self_attn_weights = (
                torch.stack(self_attn_list).mean(dim=0).to(torch.float16)
            )
        if len(cross_attn_list) > 0:
            cross_attn_weights = (
                torch.stack(cross_attn_list).mean(dim=0).to(torch.float16)
            )
        # :TODO rewrite this part logits not needed for running the other timepoints
        decoder_logits = self.decoder_fc(dec_embedding)
        if tgt_input_id is not None:
            mean_embedding = mean_nonpadding_embs(
                embs=dec_embedding,
                input_ids=tgt_input_id,
                mapping_dict=self.gene_to_rowid,
                condition_dict=self.condition_dict,
                perturbation_tokens=self.tgt_pert_tokens,
            )
        else:
            mean_embedding = None

        outputs = {
            'dec_embedding': dec_embedding,
            'self_attn_weights': self_attn_weights,
            'cross_attn_weights': cross_attn_weights,
            'dec_logits': decoder_logits,
            'labels': labels,
            'mean_embedding': mean_embedding,
        }
        return outputs
