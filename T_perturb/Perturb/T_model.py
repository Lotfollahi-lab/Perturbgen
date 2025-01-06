from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from T_perturb.Modules.T_model import CellGen
from T_perturb.src.utils import (
    generate_pad,
    gumbel_sample,
    noise_schedule,
    prob_mask_like,
    top_k,
    uniform,
)


class PerturberMasking(CellGen):
    def __init__(
        self,
        pad_token: int = 0,
        condition_dict: Dict[str, dict[str, int]] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_scheduler = kwargs['mask_scheduler']
        self.token_embedding = nn.Embedding(
            kwargs['tgt_vocab_size'], kwargs['d_model'], padding_idx=pad_token
        )
        # add <null> token for uncoditional generation
        if condition_dict is not None:
            condition_dict_ = condition_dict.copy()
            condition_dict_['uncondition'] = {'<null>': 0}
            cond_vocab_size = sum(len(v) for v in condition_dict_.values())
            self.cond_embedding = nn.Embedding(cond_vocab_size, kwargs['d_model'])

    def generate_mask(
        self,
        tgt_input_id,
        tgt_pad,
    ):
        '''
        Description:
        ------------
            Prepare masked tokens for the target input.

        Parameters:
        -----------
            tgt_input_id: `torch.Tensor`
                Target token input.
            tgt_pad: `torch.Tensor`
                Target padding mask.
            mlm_probability: `float`
                Fraction of tokens to mask.
            mask_mode: `str`
                Masking mode: ['BERT', 'MASKGIT']
                BERT: 80% MASK, 10% random, 10% original.
                MASKGIT: mask tokens based on the mask scheduler.
            mask_scheduler: `str`
                Masking scheduler defining
                the proportion of tokens to mask for MASKGIT

        Returns:
        --------
            tgt_input_id: `torch.Tensor`
                Target token input with masked tokens.
            labels: `torch.Tensor`
                True labels for masked tokens. Return -100 for non-masked tokens.
        '''
        device = tgt_input_id.device
        labels = tgt_input_id.clone()
        batch, seq_len = tgt_input_id.shape
        sample_length = torch.sum(~tgt_pad, dim=1)
        rand_time = uniform((batch,), device=device)
        rand_mask_probs = noise_schedule(
            rand_time,
            method=self.mask_scheduler,
            total_tokens=torch.tensor((seq_len), device=device),
        )
        num_token_masked = (
            (torch.mul(sample_length, rand_mask_probs)).round().clamp(min=1)
        )

        print('number of unmasked tokens', num_token_masked)
        rand_int = torch.rand((batch, seq_len), device=device)
        # exclude CLS token and set pad token to >1 to exclude from masking
        rand_int[tgt_pad] = 2
        batch_randperm = rand_int.argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        tgt_input_id = tgt_input_id.clone()
        tgt_input_id[mask] = self.mask_token
        labels[~mask] = -100
        return tgt_input_id, labels

    def forward_with_cond_scale(self, cond_scale=0.0, *args, **kwargs):
        # ---classifier-free guidance---
        # run two fwd passes with the same time step
        # one with conditional and one unconditional
        random_tps = [np.random.choice(self.pred_tps)]

        conditional_out = self.forward(
            cond_drop_prob=0.0, tgt_time_step=random_tps[0], *args, **kwargs
        )
        unconditional_out = self.forward(
            cond_drop_prob=1.0, tgt_time_step=random_tps[0], *args, **kwargs
        )
        if cond_scale == 1:
            return conditional_out
        elif cond_scale == 0:
            return unconditional_out
        else:
            for t in conditional_out.keys():
                null_logits = unconditional_out[t]['dec_logits']
                logits = conditional_out[t]['dec_logits']
                scaled_logits = null_logits + (logits - null_logits) * cond_scale
                conditional_out[t]['dec_logits'] = scaled_logits
            return conditional_out

    def forward(
        self,
        src_input_id: torch.Tensor,
        not_masked: bool = False,
        tgt_time_step: int | None = None,
        tgt_input_id_dict: dict | None = None,
        generate_id_dict: dict | None = None,
        generate_pad_dict: dict | None = None,
        condition_ids: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
    ):
        if tgt_input_id_dict:
            tgt_pad_dict = self.call_padding(
                tgt_input_id_dict,
                self.pred_tps,
            )
        else:
            tgt_pad_dict = generate_pad_dict
        src_attention_mask = generate_pad(src_input_id)
        enc_output = self.call_encoder(src_input_id, src_attention_mask)
        if (not_masked) and (tgt_input_id_dict is not None):
            # not masked for count prediction and predicted embeddings
            sorted_time_steps = sorted(self.pred_tps)
            context_time_steps = sorted_time_steps
        elif not_masked is False:
            if self.context_tps is not None:
                context_time_steps = sorted(self.context_tps)
            else:
                context_time_steps = sorted(self.pred_tps)
            if tgt_time_step is None:
                # randomly select a time step for training
                sorted_time_steps = [np.random.choice(self.pred_tps)]
            elif generate_id_dict is not None:
                # MASKGIT generation
                tgt_input_id_dict = generate_id_dict
                sorted_time_steps = [tgt_time_step]
            else:
                # cfg
                sorted_time_steps = [tgt_time_step]
        all_outputs = {}
        for tgt_time_step in sorted_time_steps:
            tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
            if tgt_input_id_dict is not None:
                tgt_input_id = tgt_input_id_dict[f'tgt_input_ids_t{tgt_time_step}']
            else:
                raise ValueError(
                    'tgt_input_id_dict or generate_id_dict must be provided'
                )
            if self.context_mode:
                # distinction between selected time step and rest time steps
                context_output, context_mask = self.generate_context(
                    enc_output=enc_output,
                    src_attention_mask=src_attention_mask,
                    tgt_time_step=tgt_time_step,
                    all_time_steps=context_time_steps,
                    tgt_input_id_dict=tgt_input_id_dict,
                    tgt_pad_dict=tgt_pad_dict,
                )
            if (not_masked is False) & (generate_id_dict is None):
                # apply masking during first stage of MLM training
                tgt_input_id, labels = self.generate_mask(
                    tgt_input_id,
                    tgt_pad,
                )
            else:
                # no true labels for MLM loss
                labels = None

            tgt_embedding = self.token_embedding(tgt_input_id)

            # ---classifier-free guidance---
            if condition_ids is not None:
                cond_token_emb = self.cond_embedding(condition_ids)
                cond_len = cond_token_emb.shape[1]
                tgt_embedding = torch.cat([cond_token_emb, tgt_embedding], dim=1)
                cond_pad = prob_mask_like(
                    condition_ids.shape, cond_drop_prob, device=condition_ids.device
                )
                tgt_pad = torch.cat([cond_pad, tgt_pad], dim=1)
                if labels is not None:
                    # add -100 to ignore condition tokens in CE loss
                    labels = F.pad(labels, (cond_len, 0), value=-100)
                    print('cond', cond_len)
                    print('labels', labels.shape)

            tgt_embedding = self.pos_embedding(tgt_embedding, tgt_time_step)
            # does not include any context
            outputs = self.call_decoder(
                enc_output=context_output if self.context_mode else enc_output,
                src_attention_mask=context_mask
                if self.context_mode
                else src_attention_mask,
                dec_embedding=tgt_embedding,
                tgt_pad=tgt_pad,
                labels=labels,
            )
            all_outputs[tgt_time_step] = outputs
        return all_outputs

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
        generate_id_dict: Dict[int, torch.Tensor] = {}
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
            ids[:, :prompt_length] = tgt_input_id[:, :prompt_length]
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
            generate_id_dict[time_step] = generated_ids
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
                # print the shape and then check genes_to_perturb
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
