'''
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from transformers import BertForMaskedLM

from T_perturb.src.utils import (
    generate_pad,
    gumbel_sample,
    noise_schedule,
    top_k,
)

# def drop_path(x, drop_prob: float = 0.0, training: bool = False):
#     if drop_prob == 0.0 or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (
#         x.ndim - 1
#     )  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output


# class DropPath(nn.Module):
#     '''
#     Drop paths (Stochastic Depth) per sample
#     (when applied in main path of residual blocks).
#     '''

#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob


#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, n_time_steps, mode='GF_fine_tuned'):
        '''
        Description:
        ------------
        Positional encoding for the transformer model.
        Two separate positional encodings are used, depending on the mode:
        - Geneformer: BERT positional encoding is from pre-trained model.
        - Transformer_encoder: Sinusoidal positional encoding is used.
        Parameters:
        -----------
        d_model: `int`
            Token embedding dimension.
        max_seq_length: `int`
            Maximum sequence length.
        n_time_steps: `int`
            Number of time steps for training.
        mode: `str`
            Mode of transformer encoder.
            Options: ['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder']
        Returns:
        --------
        x: `torch.Tensor`
            Positional embeddings.
        '''
        # train time steps and interpolation timestep
        # TODO: separate timestep positional encoding
        # and positional encoding for the ranks
        super(SinusoidalPositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        if mode in ['GF_frozen', 'GF_fine_tuned']:
            total_seq_length = n_time_steps * max_seq_length
        elif mode == 'Transformer_encoder':
            # add one time step to included src time step
            total_seq_length = (n_time_steps + 1) * max_seq_length
        self.mode = mode
        pe = torch.zeros(total_seq_length, d_model)
        position = torch.arange(0, total_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, tgt_time_step=None):
        if self.mode in ['GF_frozen', 'GF_fine_tuned']:
            tgt_time_step_ = tgt_time_step - 1
        elif self.mode == 'Transformer_encoder':
            # start from 0 to include src time step
            tgt_time_step_ = tgt_time_step
        if tgt_time_step is not None:
            start_pos = (tgt_time_step_) * self.max_seq_length
            end_pos = start_pos + x.size(1)
            pe = self.pe[:, start_pos:end_pos]
        else:
            pe = self.pe[:, : x.size(1)]
        return x + pe


# class LearntPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(LearntPositionalEncoding, self).__init__()
#         self.position_embeddings = nn.Embedding(max_seq_length, d_model)
#         self.register_buffer(
#             "position_ids", torch.arange(max_seq_length).expand((1, -1)),
#             persistent=False
#         )


#     def forward(self, x, position_ids):
#         #TODO: register buffer
#         positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)

#         return x + self.position_embeddings(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        '''
        Description:
        ------------
        Cross attention module for transformer model with two options:
        - self attention: context_dim is None
        - cross attention: context_dim is not None
        Parameters:
        -----------
        query_dim: `int`
            Query dimension.
        context_dim: `int`
            Context dimension.
        num_heads: `int`
            Number of attention heads.
        dim_head: `int`
            Dimension of the attention head.
        dropout: `float`
            Dropout rate.
        '''
        super().__init__()
        inner_dim = dim_head * num_heads
        if context_dim is None:
            context_dim = query_dim
        self.scale = dim_head**-0.5
        self.num_heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)  # projection head
        )

    def forward(self, x, context=None, mask=None):
        h = self.num_heads
        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_ff: int,
        hidden_size: int,
        dropout: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        context_dim: Optional[int] = None,
    ):
        '''
        Description:
        ------------
        Transformer block with self attention and cross attention.
        Encoder output is used as context for cross attention.
        Parameters:
        -----------
        dim: `int`
            Query dimension.
        num_heads: `int`
            Number of attention heads.
        d_ff: `int`
            Dimension of attention head.
        hidden_size: `int`
            Hidden size of the feed forward network.
        dropout: `float`
            Dropout rate.
        act_layer: `nn.Module`
            Activation layer.
        norm_layer: `nn.Module`
            Normalization layer.
        context_dim: `int`
            Context dimension for cross attention.
        Returns:
        --------
        x: `torch.Tensor`
            Output tensor.
        '''
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self attention by not passing context dim
        self.self_attn = CrossAttention(
            query_dim=dim, num_heads=num_heads, dim_head=d_ff, dropout=dropout
        )
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dim_head=d_ff,
            dropout=dropout,
        )
        self.norm3 = norm_layer(dim)
        self.feed_forward = Mlp(
            in_features=dim, hidden_features=hidden_size, act_layer=act_layer
        )  # add hidden size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, tgt_mask=None, enc_output=None):
        attn_output = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, context=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Geneformerwrapper(nn.Module):
    def __init__(
        self,
        model_path='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/Geneformer/',
        output_attentions=False,
        output_hidden_states=True,
        mode='GF_frozen',
    ):
        '''
        Description:
        ------------
        Wrapper for Geneformer model.
        Parameters:
        -----------
        model_path: `str`
            Path to the Geneformer model.
        output_attentions: `bool`
            Whether to output attentions.
        output_hidden_states: `bool`
            Whether to output hidden states.
        mode: `str`
            Mode of the Geneformer model.
            Options: ['GF_frozen', 'GF_fine_tuned']
        '''
        super(Geneformerwrapper, self).__init__()
        if mode in ['GF_frozen', 'GF_fine_tuned']:
            self.model = BertForMaskedLM.from_pretrained(
                model_path,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        self.mode = mode
        if self.mode == 'GF_frozen':
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, src_input_id, src_attention_mask):
        if self.mode == 'GF_frozen':
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=src_input_id, attention_mask=src_attention_mask
                )

        elif self.mode == 'GF_fine_tuned':
            outputs = self.model.forward(
                input_ids=src_input_id, attention_mask=src_attention_mask
            )
        embs = outputs.hidden_states[-1]
        return embs


class Encoder(nn.Module):
    '''
    Description:
    ------------
    Transformer encoder modified from
    URL: https://pytorch.org/tutorials/beginner/transformer_tutorial.html # noqa
    Last accessed: 2024-05-19
    Parameters:
    -----------
    total_vocab_size: `int`
        Total vocabulary size.
    max_seq_length: `int`
        Maximum sequence length.
    n_time_steps: `int`
        Number of time steps for positional encoding.
    d_model: `int`
        Token embedding dimension.
    nhead: `int`
        Number of attention heads.
    nlayers: `int`
        Number of attention layers.
    dropout: `float`
        Dropout rate.
    d_ff: `int`
        Dimension of the feed forward network.
    Returns:
    --------
    output: `torch.Tensor`
        Output tensor.
    '''

    def __init__(
        self,
        total_vocab_size: int,
        max_seq_length: int,
        n_time_steps: int,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 6,
        dropout: float = 0.02,
        d_ff: int = 512,
    ):
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            n_time_steps=n_time_steps,
            mode='Transformer_encoder',
        )
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            # batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_embedding = nn.Embedding(total_vocab_size, d_model, padding_idx=0)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        '''
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, total_vocab_size]``
        '''
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(x=src, tgt_time_step=0)
        # transpose src:
        # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        # reverse transpose
        output = output.transpose(0, 1)
        return output

    # def forward(
    #         self,
    #         src: torch.Tensor,
    #         src_mask: torch.Tensor = None
    #         ) -> torch.Tensor:
    #     '''
    #     Parameters:
    #     -----------
    #     src: `torch.Tensor`
    #         shape ``[batch_size, seq_len, total_vocab_size]``
    #     src_mask: `torch.Tensor`
    #         shape ``[batch_size, seq_len]``
    #     Returns:
    #     --------
    #     output: `torch.Tensor`
    #         shape ``[batch_size, seq_len, total_vocab_size]``
    #     '''
    #     src = self.token_embedding(src) * math.sqrt(self.d_model)
    #     src = self.positional_encoding(x=src, tgt_time_step=0)
    #     output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
    #     return output


class CellGen(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int = 25426,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        d_ff: int = 2048,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        mlm_probability: float = 0.3,
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
        mode='GF_frozen',
    ):
        '''
        Description:
        ------------
        Seq2Seq model for cell generation
        using masked language modeling adopted from MaskGIT.
        Parameters:
        -----------
        tgt_vocab_size: `int`
            Target vocabulary size.
        d_model: `int`
            Token embedding dimension.
        num_heads: `int`
            Number of attention heads.
        num_layers: `int`
            Number of attention layers.
        d_ff: `int`
            Dimension of the feed forward network.
        max_seq_length: `int`
            Maximum sequence length.
        dropout: `float`
            Dropout rate.
        mlm_probability: `float`
            Fraction of tokens to mask.
        time_steps: `list`
            List of time steps for training and testing.
        total_time_steps: `int`
            Total number of time steps.
        mode: `str`
            Mode of the encoder.
            Options: ['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder']
        Returns:
        --------
        outputs: `dict`
            Output dictionary containing the following keys:
            - 'dec_logits': Decoder logits.
            - 'labels': True labels for masked tokens.
            - 'selected_time_step': Selected time step.
            - 'dec_embedding': Decoder embeddings.
            - 'mean_embedding': Mean embeddings for non-padding tokens.
            - 'cls_positions': CLS token positions.
        '''
        super(CellGen, self).__init__()
        self.num_features = self.embed_dim = d_model
        self.mlm_probability = mlm_probability
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        total_vocab_size = (
            tgt_vocab_size + total_time_steps
        )  # add one for each cls token
        self.time_steps = time_steps
        self.total_time_steps = list(range(1, total_time_steps + 1))
        self.mask_token = total_vocab_size
        total_vocab_size = total_vocab_size + 1  # add one for padding token
        self.token_embedding = nn.Embedding(total_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            n_time_steps=total_time_steps,
            mode=mode,
        )
        if mode in ['GF_frozen', 'GF_fine_tuned']:
            self.encoder_layers = Geneformerwrapper(mode=mode)
        elif mode == 'Transformer_encoder':
            self.encoder_layers = Encoder(
                total_vocab_size=total_vocab_size,
                max_seq_length=max_seq_length,
                n_time_steps=total_time_steps,
                d_model=d_model,
            )
        else:
            raise ValueError(f'Invalid encoder mode: {mode}')
        self.mode = mode

        self.decoder_block = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    hidden_size=d_model,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_fc = nn.Linear(d_model, tgt_vocab_size)  # Specify the GPU device)
        self.dropout = nn.Dropout(dropout)

    #     self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.token_embedding.weight.data.uniform_(-initrange, initrange)

    def mean_nonpadding_embs(self, embs, pad, dim=1):
        '''
        Compute the mean of the non-padding embeddings.
        Modified from Geneformer:
        https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py # noqa
        Accessed: 2024-05-14
        '''
        # mask should be opposite of pad
        pad[:, 0] = True
        # our mask is the opposite of BERT mask
        pad_mask = ~pad
        # create a tensor of original lengths
        original_lens = pad_mask.sum(dim=1)

        # create CLS token mask
        if embs.dim() == 3:
            # fill the masked positions in embs with zeros
            masked_embs = embs.masked_fill(~pad_mask.unsqueeze(2), 0.0)

            # compute the mean across the non-padding dimensions
            mean_embs = masked_embs.sum(dim) / original_lens.view(-1, 1).float()

        elif embs.dim() == 2:
            masked_embs = embs.masked_fill(~pad_mask, 0.0)
            mean_embs = masked_embs.sum(dim) / original_lens.float()
        return mean_embs

    def generate_mask(self, tgt_input_id, tgt_pad, mlm_probability=0.15):
        '''
        Description:
        ------------
        Masked language modeling for the target tokens.
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Modified from Huggingface Transformers library:
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L840 # noqa
        Accessed: 2024-05-12
        Parameters:
        -----------
        tgt_input_id: `torch.Tensor`
            Target token input.
        tgt_pad: `torch.Tensor`
            Target padding mask.
        mlm_probability: `float`
            Fraction of tokens to mask.
        Returns:
        --------
        tgt_input_id: `torch.Tensor`
            Target token input with masked tokens.
        labels: `torch.Tensor`
            True labels for masked tokens. Return -100 for non-masked tokens.
        '''
        device = tgt_input_id.device
        labels = tgt_input_id.clone()
        probability_matrix = torch.full_like(
            tgt_pad, mlm_probability, dtype=torch.float
        )
        # Do not mask CLS and PAD tokens
        cls_tgt_pad = tgt_pad.clone()
        cls_tgt_pad[:, 0] = True
        probability_matrix = probability_matrix.masked_fill(cls_tgt_pad, 0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # replace 80% of the tokens with mask token,
        # 10% with random token, 10% with original token
        indices_masked = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool()
            & masked_indices
        )
        tgt_input_id[masked_indices] = self.mask_token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
            & masked_indices
            & ~indices_masked
        )
        # +1 to exclude pad and cls tokens from random token selection
        random_tokens = torch.randint(
            1, self.tgt_vocab_size, labels.shape, dtype=torch.long, device=device
        )
        tgt_input_id[indices_random] = random_tokens[indices_random]
        return tgt_input_id, labels

    def call_padding(
        self,
        tgt_input_id_dict,
        time_steps,
    ):
        tgt_pad_dict = {}
        for time_step in time_steps:
            tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']
            tgt_pad_dict[f'tgt_pad_t{time_step}'] = generate_pad(tgt_input_id)
        return tgt_pad_dict

    def call_encoder(self, src_input_id, src_attention_mask):
        if self.mode in ['GF_frozen', 'GF_fine_tuned']:
            # BERT mask: 1 for tokens to keep, 0 for tokens to mask. Thus, negate mask.
            src_attention_mask = ~src_attention_mask
            enc_output = self.encoder_layers(src_input_id, src_attention_mask.int())
        else:
            # different mask for transformer encoder
            enc_output = self.encoder_layers(src_input_id, src_attention_mask)
        return enc_output

    def call_decoder(
        self,
        enc_output,
        src_attention_mask,
        dec_embedding,
        tgt_pad,
        time_random,
        generate=False,
        labels=None,
        cls_positions=None,
    ):
        for dec_layer in self.decoder_block:
            # see if concatenation of cls embedding
            dec_embedding = dec_layer(
                x=dec_embedding,
                src_mask=src_attention_mask,
                tgt_mask=tgt_pad,
                enc_output=enc_output,
            )
        # :TODO rewrite this part logits not needed for running the other timepoints
        outputs = {}
        decoder_logits = self.decoder_fc(dec_embedding)
        if labels is not None:
            outputs['dec_logits'] = decoder_logits
            outputs['labels'] = labels
            outputs['selected_time_step'] = time_random
        if generate is True:
            outputs['dec_embedding'] = dec_embedding
            outputs['mean_embedding'] = self.mean_nonpadding_embs(
                embs=dec_embedding,
                pad=tgt_pad,
            )
            outputs['dec_logits'] = decoder_logits[:, 1:, :]
        else:
            outputs['dec_embedding'] = dec_embedding
            outputs['mean_embedding'] = self.mean_nonpadding_embs(
                embs=dec_embedding,
                pad=tgt_pad,
            )

        if cls_positions is not None:
            outputs['cls_positions'] = cls_positions
        return outputs

    def generate_context(
        self,
        enc_output,
        src_attention_mask,
        tgt_time_step,
        all_time_steps,
        tgt_input_id_dict,
        tgt_pad_dict,
        cls_positions=None,
        generate=False,
    ):
        context_embedding_dict = {}
        context_embedding_dict['context_t0'] = enc_output
        context_pad_dict = {}
        context_pad_dict['src_pad'] = src_attention_mask

        # retrieve the embeddings to provide as context
        # pad the rest of the time steps
        for time_step in all_time_steps:
            # exclude tgt_time_step from the context
            if time_step != tgt_time_step:
                # if (generate is True) and (time_step in self.time_steps):
                # only provide previous time steps as context
                if len(context_embedding_dict) > 1:
                    context_tensors = list(context_embedding_dict.values())
                    context = torch.cat(context_tensors, dim=1)
                else:
                    context = enc_output
                context_pads = list(context_pad_dict.values())
                context_pad = torch.cat(context_pads, dim=1)
                tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']
                tgt_pad = tgt_pad_dict[f'tgt_pad_t{time_step}']
                if not generate:
                    tgt_input_id = tgt_input_id.masked_fill(tgt_pad, 0)
                with torch.no_grad():
                    tgt_embedding = self.token_embedding(tgt_input_id)
                    dec_embedding = self.positional_encoding(tgt_embedding, time_step)

                    # create context for the ones before the selected time step
                    # pad the rest
                    dec_outputs = self.call_decoder(
                        enc_output=context,
                        src_attention_mask=context_pad,
                        dec_embedding=dec_embedding,
                        tgt_pad=tgt_pad,
                        time_random=time_step,
                        generate=generate,
                        labels=None,
                        cls_positions=cls_positions,
                    )
                context_embedding_dict[f'context_t{time_step}'] = dec_outputs[
                    'dec_embedding'
                ]
                # append the current pad to the previous context pad
                context_pad_dict[f'tgt_pad_t{time_step}'] = tgt_pad
        return context_embedding_dict, context_pad_dict

    def context_backprop(
        self,
        context_embedding_dict,
        context_pad_dict,
        tgt_pad,
        tgt_time_step,
        tgt_input_id_dict,
        cls_positions=None,
        generate=False,
        not_masked=False,
    ):
        # selected_tgt_pad = context_pad_dict[f'tgt_pad_t{tgt_time_step}']
        selected_tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{tgt_time_step}']
        context_embedding_dict_ = context_embedding_dict.clone()
        context_pad_dict_ = context_pad_dict.clone()
        # remove subsequent time steps from context and pad
        # if not generate:
        #     for time_step in self.time_steps:
        #         if time_step >= tgt_time_step:
        #             context_embedding_dict_.pop(f'context_t{time_step}')
        #             context_pad_dict_.pop(f'tgt_pad_t{time_step}')
        # # selected time step should not be included in the context for generation
        # if generate is False:
        #     context_embedding_dict_.pop(f'context_t{tgt_time_step}')
        if generate:
            context_pad_dict_.pop(f'tgt_pad_t{tgt_time_step}')
        context_tensors = list(context_embedding_dict_.values())

        context_embedding = torch.cat(context_tensors, dim=1)
        context_pads = list(context_pad_dict_.values())
        context_pad = torch.cat(context_pads, dim=1)
        # only create maskings for the selected time step
        if not_masked or generate:
            labels = None
            # do not mask for embeddings for testing
            masked_tgt_input_id = selected_tgt_input_id
        else:
            masked_tgt_input_id, labels = self.generate_mask(
                selected_tgt_input_id,
                tgt_pad,
                self.mlm_probability,
            )
        selected_tgt_embedding = self.token_embedding(masked_tgt_input_id)

        selected_tgt_embedding = self.positional_encoding(
            selected_tgt_embedding, tgt_time_step
        )
        outputs = self.call_decoder(
            enc_output=context_embedding,
            src_attention_mask=context_pad,
            dec_embedding=selected_tgt_embedding,
            tgt_pad=tgt_pad,
            time_random=tgt_time_step,
            generate=generate,
            labels=labels,
            cls_positions=cls_positions,
        )
        return outputs

    def forward(
        self,
        src_input_id: torch.Tensor,
        cls_positions: torch.Tensor,
        not_masked: bool = False,
        context_mode: bool = False,
        tgt_time_step: Optional[int] = None,
        tgt_input_id_dict: Optional[dict] = None,
        generate_id_dict: Optional[dict] = None,
        generate_pad_dict: Optional[dict] = None,
    ):
        '''
        Description:
        ------------
        Forward pass for the Seq2Seq model.
        Parameters:
        -----------
        src_input_id: `torch.Tensor`
            Source token input.
        tgt_time_step: `int`
            Target time step.
        cls_positions: `torch.Tensor`
            CLS token positions.
        not_masked: `bool`
            Whether to mask tokens. Should not be masked for testing and generation.
        context_mode: `bool`
            Whether to use context mode, where other time steps are used as context.
        generate_id_dict: `Optional[dict]`
            Dictionary of target token inputs for generation.
        generate_pad_dict: `Optional[dict]`
            Dictionary of target padding masks for generation.
        tgt_input_id_dict: `Optional[dict]`
            Dictionary of target token inputs from different time steps.
        Returns:
        --------
        outputs: `dict`
            Output dictionary
        '''
        if tgt_input_id_dict:
            tgt_pad_dict = self.call_padding(
                tgt_input_id_dict,
                self.time_steps,
            )
        else:
            tgt_pad_dict = generate_pad_dict

        src_attention_mask = generate_pad(src_input_id)
        # BERT mask: 1 for tokens to keep, 0 for tokens to mask. Thus, negate mask.
        enc_output = self.call_encoder(src_input_id, src_attention_mask)
        if context_mode:
            # distinction between selected time step and rest time steps
            if not_masked:
                dec_embedding_list = []
                mean_embedding_dict = {}
                for tgt_time_step in self.time_steps:
                    tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
                    context_embedding_dict, context_pad_dict = self.generate_context(
                        enc_output=enc_output,
                        src_attention_mask=src_attention_mask,
                        tgt_time_step=tgt_time_step,
                        all_time_steps=self.time_steps,
                        tgt_input_id_dict=tgt_input_id_dict,
                        tgt_pad_dict=tgt_pad_dict,
                        cls_positions=cls_positions,
                    )

                    # context should be all the ones before the selected time step
                    outputs = self.context_backprop(
                        context_embedding_dict=context_embedding_dict,
                        context_pad_dict=context_pad_dict,
                        tgt_pad=tgt_pad,
                        tgt_time_step=tgt_time_step,
                        tgt_input_id_dict=tgt_input_id_dict,
                        cls_positions=cls_positions,
                        not_masked=not_masked,
                    )
                    dec_embedding_list.append(outputs['dec_embedding'])
                    mean_embedding_dict[tgt_time_step] = outputs['mean_embedding']
                outputs['mean_embedding'] = mean_embedding_dict
                outputs['dec_embedding'] = torch.cat(dec_embedding_list, dim=1)

            else:
                if tgt_time_step is None:
                    tgt_time_step = np.random.choice(self.time_steps)
                    generate = False
                    context_time_steps = self.time_steps
                else:
                    tgt_input_id_dict = generate_id_dict
                    generate = True
                    context_time_steps = self.total_time_steps
                # only extract context for all the ones before the selected time step
                # rest will be padded
                # ---Initialise the decoder embeddings
                # to provide as context for selected time step---
                tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
                context_embedding_dict, context_pad_dict = self.generate_context(
                    enc_output=enc_output,
                    src_attention_mask=src_attention_mask,
                    tgt_time_step=tgt_time_step,
                    all_time_steps=context_time_steps,
                    tgt_input_id_dict=tgt_input_id_dict,
                    tgt_pad_dict=tgt_pad_dict,
                    cls_positions=cls_positions,
                    generate=generate,
                )
                if generate:
                    context_pad_dict = generate_pad_dict

                outputs = self.context_backprop(
                    context_embedding_dict=context_embedding_dict,
                    context_pad_dict=context_pad_dict,
                    tgt_pad=tgt_pad,
                    tgt_time_step=tgt_time_step,
                    tgt_input_id_dict=tgt_input_id_dict,
                    cls_positions=cls_positions,
                    not_masked=not_masked,
                    generate=generate,
                )
        else:
            if not_masked:
                dec_embedding_list = []
                mean_embedding_dict = {}
                for tgt_time_step in self.time_steps:
                    if tgt_input_id_dict is not None:
                        tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
                        tgt_input_id = tgt_input_id_dict[
                            f'tgt_input_id_t{tgt_time_step}'
                        ]
                        tgt_input_id = tgt_input_id.masked_fill(tgt_pad, 0)
                        tgt_embedding = self.token_embedding(tgt_input_id)
                        tgt_embedding = self.positional_encoding(
                            tgt_embedding, tgt_time_step
                        )
                        outputs = self.call_decoder(
                            enc_output=enc_output,
                            src_attention_mask=src_attention_mask,
                            dec_embedding=tgt_embedding,
                            tgt_pad=tgt_pad,
                            time_random=tgt_time_step,
                            generate=False,
                            labels=None,
                            cls_positions=cls_positions,
                        )
                        dec_embedding_list.append(outputs['dec_embedding'])
                        mean_embedding_dict[tgt_time_step] = outputs['mean_embedding']
                outputs['mean_embedding'] = mean_embedding_dict
                outputs['dec_embedding'] = torch.cat(dec_embedding_list, dim=1)
            else:
                if tgt_time_step is None:
                    if tgt_input_id_dict is not None:
                        tgt_time_step = np.random.choice(self.time_steps)
                        generate = False
                        tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
                        tgt_input_id = tgt_input_id_dict[
                            f'tgt_input_id_t{tgt_time_step}'
                        ]
                        tgt_input_id = tgt_input_id.masked_fill(tgt_pad, 0)
                        tgt_input_id, labels = self.generate_mask(
                            tgt_input_id,
                            tgt_pad,
                            self.mlm_probability,
                        )
                else:
                    if generate_id_dict is not None:
                        tgt_input_id_dict = generate_id_dict
                        generate = True
                        tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
                        labels = None
                        tgt_input_id = tgt_input_id_dict[
                            f'tgt_input_id_t{tgt_time_step}'
                        ]
                tgt_embedding = self.token_embedding(tgt_input_id)
                tgt_embedding = self.positional_encoding(tgt_embedding, tgt_time_step)
                outputs = self.call_decoder(
                    enc_output=enc_output,
                    src_attention_mask=src_attention_mask,
                    dec_embedding=tgt_embedding,
                    tgt_pad=tgt_pad,
                    time_random=tgt_time_step,
                    generate=generate,
                    labels=labels,
                    cls_positions=cls_positions,
                )
        return outputs


class CountHead(nn.Module):
    def __init__(
        self,
        loss_mode: str = 'zinb',
        tgt_vocab_size: int = 25426,
        d_model: int = 256,
        dropout: float = 0.0,
    ):
        super(CountHead, self).__init__()
        self.loss_mode = loss_mode

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=d_model,
            drop=dropout,
        )
        n_genes = tgt_vocab_size
        if self.loss_mode == 'mse':
            self.relu_output = nn.Sequential(nn.Linear(d_model, n_genes), nn.ReLU())
        elif self.loss_mode == 'zinb':
            self.linear_output = nn.Linear(d_model, n_genes)
            self.softmax_output = nn.Sequential(
                nn.Linear(d_model, n_genes), nn.Softmax(dim=-1)
            )
        elif self.loss_mode == 'nb':
            self.softmax_output = nn.Sequential(
                nn.Linear(d_model, n_genes), nn.Softmax(dim=-1)
            )

    def forward(self, x):
        # use cls token for count prediction
        count_outputs = {}
        mlp_output = self.mlp(x)
        mlp_output = nn.functional.normalize(mlp_output, dim=-1, p=2)
        if self.loss_mode == 'mse':
            count_outputs['count_lognorm'] = self.relu_output(mlp_output)
        elif self.loss_mode == 'zinb':
            count_outputs['count_mean'] = self.softmax_output(mlp_output)
            count_outputs['count_dropout'] = self.linear_output(mlp_output)
        elif self.loss_mode == 'nb':
            count_outputs['count_mean'] = self.softmax_output(mlp_output)
        return count_outputs


class CountDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module = None,
        loss_mode: str = 'zinb',
        tgt_vocab_size: int = 25426,
        d_model: int = 256,
        add_mask_id: bool = True,
        dropout: float = 0.0,
        time_steps: list = [1, 2],
        total_time_steps: int = 3,
    ):
        super(CountDecoder, self).__init__()
        self.pretrained_model = pretrained_model
        self.embed_dim = d_model

        self.loss_mode = loss_mode
        # exclude pad (-1) to get the number of genes
        self.count_decoder = CountHead(loss_mode, tgt_vocab_size - 1, d_model, dropout)
        total_vocab_size = tgt_vocab_size + total_time_steps  # add one for cls token
        if add_mask_id:
            self.mask_token = total_vocab_size

        self.time_steps = time_steps
        self.total_time_steps = list(range(1, total_time_steps + 1))
        self.cls_embedding = None

    def generate_pad(self, tgt):
        tgt_pad = tgt == 0
        return tgt_pad

    def forward(
        self,
        src_input_id,
        tgt_input_id_dict,
        cls_positions=None,
    ):
        outputs = self.pretrained_model(
            src_input_id=src_input_id,
            tgt_input_id_dict=tgt_input_id_dict,
            not_masked=True,
            cls_positions=cls_positions,
        )

        count_outputs = {}
        for _, t in enumerate(self.time_steps):
            # cls_position = cls_positions[i]
            # cls_embedding = outputs['dec_embedding'][:, cls_position, :]
            # print("cls_embedding",cls_embedding.shape)
            cls_embedding = outputs['mean_embedding'][t]
            # print("cls_embedding",cls_embedding.shape)
            count_outputs_tmp = self.count_decoder.forward(cls_embedding)
            count_outputs[f'count_output_t{t}'] = count_outputs_tmp

        return count_outputs

    def call_padding(self, src_input_id, tgt_input_id_dict, time_steps):
        tgt_pad_dict = {}
        src_attention_mask = self.generate_pad(src_input_id)
        tgt_pad_dict['src_pad'] = src_attention_mask
        for time_step in time_steps:
            tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']
            tgt_pad_dict[f'tgt_pad_t{time_step}'] = self.generate_pad(tgt_input_id)
        return tgt_pad_dict

    def generate(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id_dict: dict,
        max_len: int,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        # time_steps=[1, 2, 3],
        temperature: float = 2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        iterations: int = 18,  # optimal of iterations in MaskGIT
        mask_scheduler: str = 'cosine',
        cls_positions=[0, 247, 494],
    ):
        '''
        Description:
        ------------
        Generate sequences for the target tokens
        adopted from MaskGIT using the pretrained model.
        Use mean non-padding embeddings for count prediction.
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
        Returns:
        --------
        count_outputs: `dict`
            Output dictionary containing the following keys:
            - 'count_output_t{t}': Count prediction for time step t.
            - 'cls_embedding_t{t}': CLS token embeddings for time step t.
        '''
        starting_temperature = temperature
        demask_fn = self.pretrained_model
        generate_id_dict = {}
        generate_pad_dict = {}
        dec_embedding_list = []
        for i, time_step in enumerate(self.time_steps):
            generate_id_dict = tgt_input_id_dict.copy()
            generate_pad_dict = self.call_padding(
                src_input_id, generate_id_dict, self.total_time_steps
            )
            # use max shape instead of genes you like to generate
            pad_tensor = torch.ones_like(generate_pad_dict[f'tgt_pad_t{time_step}'])
            if pad_tensor.shape[1] > max_len:
                # set the rest of the tokens to zero
                pad_tensor[:, max_len:] = 0
            tgt_pad = self.generate_pad(pad_tensor)
            generate_pad_dict[f'tgt_pad_t{time_step}'] = tgt_pad
            tgt_input_id_key = f'tgt_input_id_t{time_step}'
            tgt_input_id = tgt_input_id_dict[tgt_input_id_key]
            batch_size = tgt_input_id.shape[0]
            seq_len = tgt_input_id.shape[1]
            shape = (batch_size, seq_len)
            # create ids and scores matrix for each batch
            ids = torch.full(
                shape, self.mask_token, dtype=torch.long, device=tgt_input_id.device
            )
            # add cls token to the ids
            ids[:, 0] = tgt_input_id[:, 0]
            generate_id_dict[tgt_input_id_key] = ids
            # pad ids
            scores = torch.zeros(shape, dtype=torch.float, device=tgt_input_id.device)
            outputs, generated_ids = self.generate_sequence(
                generate_id_dict=generate_id_dict,
                generate_pad_dict=generate_pad_dict,
                src_input_id=src_input_id,
                demask_fn=demask_fn,
                mask_scheduler=mask_scheduler,
                can_remask_prev_masked=can_remask_prev_masked,
                topk_filter_thres=topk_filter_thres,
                starting_temperature=starting_temperature,
                iterations=iterations,
                scores=scores,
                tgt_time_step=time_step,
                cls_positions=cls_positions,
            )
            generate_id_dict[tgt_input_id_key] = generated_ids
            dec_embedding_list.append(outputs['dec_embedding'])
        outputs['dec_embedding'] = torch.cat(dec_embedding_list, dim=1)
        count_outputs = {}
        for i, t in enumerate(self.time_steps):
            cls_embedding = outputs['mean_embedding']
            count_outputs_tmp = self.count_decoder.forward(cls_embedding)
            count_outputs[f'count_output_t{t}'] = count_outputs_tmp
            count_outputs[f'cls_embedding_t{t}'] = cls_embedding
        return count_outputs

    def generate_sequence(
        self,
        generate_id_dict: dict,
        generate_pad_dict: dict,
        src_input_id: torch.Tensor,
        demask_fn: nn.Module,
        mask_scheduler: str,
        scores: torch.Tensor,
        cls_positions: torch.tensor,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        starting_temperature: float = 2.0,
        iterations: int = 18,
        tgt_time_step: Optional[int] = 1,
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
        outputs: `dict`
            Output dictionary containing the following keys:
            - 'dec_logits': Decoder logits.
            - 'labels': True labels for masked tokens.
            - 'selected_time_step': Selected time step.
            - 'dec_embedding': Decoder embeddings.
            - 'mean_embedding': Mean embeddings for non-padding tokens.
            - 'cls_positions': CLS token positions.
        tmp_ids: `torch.Tensor`
            Generated target token inputs.
        '''
        for iteration, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, iterations),
                reversed(range(iterations)),
            ),
            total=iterations,
        ):
            tmp_ids = generate_id_dict[f'tgt_input_id_t{tgt_time_step}']
            tgt_pad = generate_pad_dict[f'tgt_pad_t{tgt_time_step}']
            cls_token = tmp_ids[:, 0]
            # mask scheduler function, gamma
            rand_mask_prob = noise_schedule(
                ratio=iteration,
                total_tokens=tmp_ids.shape[1],
                method=mask_scheduler,
            )
            scores = scores.masked_fill(tgt_pad, -torch.finfo().max)
            tmp_ids = tmp_ids.masked_fill(tgt_pad, 0)
            ids_to_keep = torch.zeros_like(tmp_ids, dtype=torch.long)

            for i, score in enumerate(scores):
                # count zeros in each row
                unpadded = len(score) - sum(score == -torch.finfo().max)
                num_token_masked = int(unpadded * rand_mask_prob)
                masked_indices = score.topk(num_token_masked, dim=-1).indices
                mask = torch.zeros_like(tmp_ids[i], dtype=torch.bool)
                mask[masked_indices] = True
                tmp_ids[i, masked_indices] = self.mask_token
                # keep indices which are not masked
                ids_to_keep[i, ~mask] = tmp_ids[i, ~mask]

            tmp_ids[:, 0] = cls_token
            generate_id_dict[f'tgt_input_id_t{tgt_time_step}'] = tmp_ids
            outputs = demask_fn.forward(
                src_input_id=src_input_id,  # target
                generate_id_dict=generate_id_dict,
                generate_pad_dict=generate_pad_dict,
                not_masked=False,
                tgt_time_step=tgt_time_step,
                cls_positions=cls_positions,
            )
            logits = outputs['dec_logits']
            # exclude cls token
            tmp_ids_ = tmp_ids[:, 1:]
            scores_ = scores[:, 1:]
            ids_to_keep_ = ids_to_keep[:, 1:]
            # Create a mask of already predicted tokens
            for sample in range(logits.shape[0]):
                unique_ids = torch.unique(ids_to_keep_[sample])
                logits[sample, :, unique_ids] = -float('inf')
            filtered_logits = top_k(logits, topk_filter_thres)
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
                dtype = scores_.dtype
                scores_ = scores_.masked_fill(~is_mask, -torch.finfo(dtype).max)
            # add cls token to the ids and update scores and ids
            scores[:, 1:] = scores_
            tmp_ids[:, 1:] = tmp_ids_

        return outputs, tmp_ids
