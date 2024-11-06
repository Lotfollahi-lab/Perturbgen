'''
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.functional import F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertForMaskedLM

from geneformer.emb_extractor import get_embs

from T_perturb.gene_embedding import extract_gene_embeddings

from T_perturb.src.utils import (
    generate_padding,
    mean_nonpadding_embs,
    noise_schedule,
    uniform,
    return_gene_embeddings,
    top_k,
    gumbel_sample,
)

# from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# from datetime import datetime

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


# class SinusoidalPositionalEncoding(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         max_seq_length: int,
#     ):
#         '''
#         Description:
#         ------------
#         Positional encoding for the transformer model.

#         Parameters:
#         -----------
#         d_model: `int`
#             Token embedding dimension.
#         max_seq_length: `int`
#             Maximum sequence length.


#         Returns:
#         --------
#         x: `torch.Tensor`
#             Positional embeddings.
#         '''
#         # train time steps and interpolation timestep
#         # TODO: separate timestep positional encoding
#         # and positional encoding for the ranks
#         super(SinusoidalPositionalEncoding, self).__init__()
#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, : x.size(1)]


# class LearntPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(LearntPositionalEncoding, self).__init__()
#         self.position_embeddings = nn.Embedding(max_seq_length, d_model)
#         # Register a buffer for position IDs,
#         # precomputed for the maximum sequence length
#         position_ids = torch.arange(max_seq_length).expand((1, -1))
#         self.register_buffer('position_ids', position_ids)

#     def forward(self, x, position_ids=None):
#         # TODO: register buffer
#         if position_ids is None:
#             position_ids = self.position_ids[:, : x.size(1)]
#         position_ids = position_ids.expand(x.size(0), -1)

#         return x + self.position_embeddings(position_ids)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
        else:
            position = position_ids.float().unsqueeze(2)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float() * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return x + pe

    
# class LearntPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(LearntPositionalEncoding, self).__init__()
#         self.position_embeddings = nn.Embedding(max_seq_length, d_model)

#     def forward(self, x, position_ids=None):
#         if position_ids is None:
#             batch_size, seq_length = x.size(0), x.size(1)
#             position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
#             position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
#         position_embeddings = self.position_embeddings(position_ids)
#         return x + position_embeddings
    

class LearntPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearntPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        # Register a buffer for position IDs,
        # precomputed for the maximum sequence length
        position_ids = torch.arange(max_seq_length).expand((1, -1))
        self.register_buffer('position_ids', position_ids)

    def forward(self, x, position_ids=None):
        # TODO: register buffer
        if position_ids is None:
            position_ids = self.position_ids[:, : x.size(1)]
        position_ids = position_ids.expand(x.size(0), -1)

        return x + self.position_embeddings(position_ids)
    
    

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

    def normal_attention(self, q, k, v, h, mask=None):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return out

    def sdpa_attention(self, q, k, v, h, mask=None):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        if mask is not None:
            # Expand the mask to match the target shape:
            # [batch_size, num_heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, 8, 2048, 2048)
            # negate mask so that padding tokens=False
            mask = ~mask
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                is_causal=False,
            )
        out = rearrange(out, ' b h n d -> b n (h d)', h=h)
        return out

    def forward(self, x, context=None, mask=None, attention_mode='sdpa'):
        h = self.num_heads
        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)
        if (x.dtype == torch.float16) or (x.dtype == torch.bfloat16):
            if attention_mode == 'normal':
                out = self.normal_attention(q, k, v, h, mask)
            elif attention_mode == 'sdpa':
                out = self.sdpa_attention(q, k, v, h, mask)
            else:
                raise ValueError(f'Invalid attention mode: {attention_mode}')
        else:
            out = self.normal_attention(q, k, v, h, mask)
        return self.to_out(out)


class FeedForward(nn.Module):

    '''
    Description:
    ------------
    Feed forward network for MoE based transformers.
    Adopted from the Mistral repository:
    URL: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer.py # noqa
    Accessed: 2024-07-17
    '''

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))  # type: ignore


class SoftGatingMoE(nn.Module):
    '''
    Description:
    ------------
    Soft gating mechanism for feed forward network layers of transformers.

    Parameters:
    -----------
    dim: `int`
        Query dimension.
    num_experts: `int`
        Number of experts.
    num_classes: `int`
        Number of classes.
    top_k: `int`
        Number of top experts to keep.
    dropout: `float`
        Dropout rate.

    Returns:
    --------
    attn_output: `torch.Tensor`
        Output tensor.
    moe_embs_dict: `dict`
        Dictionary of expert embeddings aggregated by mean non-padding tokens.
    '''

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_classes: int,
        top_k: int,
        hidden_size: int = 64,
        moe_type: Literal['moe_attention', 'none', 'moe_ffn'] = 'none',
        num_heads: int = 8,
        # d_ff: int = 1024,
        d_ff: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.moe_type = moe_type
        # Initialize experts
        if moe_type == 'moe_ffn':
            self.experts = nn.ModuleList(
                [
                    FeedForward(
                        dim=dim,
                        hidden_dim=hidden_size,
                    )
                    for _ in range(num_experts)
                ]
            )
        elif moe_type == 'moe_attention':
            self.experts = nn.ModuleList(
                [
                    CrossAttention(
                        query_dim=dim,
                        num_heads=num_heads,
                        dim_head=d_ff,
                        dropout=dropout,
                        context_dim=None,
                    )
                    for _ in range(num_experts)
                ]
            )
        self.classifier = nn.Linear(dim, 1)
        self.jitter_noise = 0.0
        # learn gate weights one on batch and one on token level
        # :TODO: set bias of token layer to false
        self.token_gating_layer = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,
        x,
        tgt_pad,
        task_categories=None,
        tgt_mask_id_bool=None,
    ):
        '''
        Adapted from the Mistral huggingface repository:
        URL:
        https://github.com/huggingface/transformers/blob/66bc4def9505fa7c7fe4aa7a248c34a026bb552b/src/transformers/models/mixtral/modeling_mixtral.py#L681 # noqa
        Accessed: 2024-09-07
        '''
        batch_size, sequence_length, hidden_dim = x.shape
        temperature = 1.0
        # Filter to keep only the top-k experts active
        if self.training and self.jitter_noise > 0:
            x *= torch.empty_like(x).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        # keep track of CLS token for each batch
        cls_mask = torch.zeros_like(tgt_pad, dtype=torch.bool)
        cls_mask[:, 0] = True
        x = x.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        gate_logits = self.token_gating_layer(x) / temperature
        # gaussian_noise = Normal(self.mean, self.std)
        # noise = gaussian_noise.sample(gate_logits.shape).squeeze(-1)
        # gate_logits = gate_logits + noise
        # gate_logits = gate_logits.masked_fill(
        #     tgt_mask_id_bool.unsqueeze(-1), -torch.finfo(gate_logits.dtype).max
        # )

        # # apply padding mask
        # gate_logits = gate_logits.masked_fill(
        #     tgt_pad.unsqueeze(-1),
        #     -torch.finfo().max
        #     )
        # add temperature scaling for the logits to make the distribution smoother

        routing_weights = F.softmax(gate_logits, dim=1, dtype=torch.float)
        # save routing weights for visualization
        routing_weights_ = routing_weights.view(
            batch_size, sequence_length, self.num_experts
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # print('top k values', routing_weights[:20,:])
        # print(routing_weights.shape)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)
        # Store logits for each expert separately in a list
        expert_logits_list = [
            torch.zeros(x.size(0), 1, device=x.device) for _ in range(self.num_experts)
        ]

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_classes
        ).permute(
            2, 1, 0
        )  # 🔍 for compatibility of different gate size
        # print(expert_mask.shape)
        # print('expert mask', expert_mask)

        # Gather the top-k expert indices and their corresponding weights
        for expert_idx in range(0, len(self.experts)):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            if self.moe_type == 'moe_ffn':
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )
            else:
                current_hidden_states = (
                    expert_layer(current_state, mask=tgt_pad)
                    * routing_weights[top_x, idx, None]
                )
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
            weighted_output = final_hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )
            expert_logits_list[expert_idx] = self.classifier(
                weighted_output[:, 0, :].clone()
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return (final_hidden_states, gate_logits, expert_logits_list, routing_weights_)


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
        num_experts: int = 2,
        num_classes: int = 2,
        top_k: int = 2,
        mode: Literal['moe_attention', 'none', 'moe_ffn'] = 'none',
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
        self.mode = mode
        self.norm1 = norm_layer(dim)

        if self.mode == 'moe_attention':
            self.moe_attn = SoftGatingMoE(
                dim=dim,
                num_experts=num_experts,
                num_classes=num_classes,
                num_heads=num_heads,
                d_ff=d_ff,
                top_k=top_k,
                dropout=dropout,
                moe_type=mode,
            )
        else:
            self.self_attn = CrossAttention(
                query_dim=dim,
                context_dim=None,
                num_heads=num_heads,
                dim_head=d_ff,
                dropout=dropout,
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
        if self.mode == 'moe_ffn':
            self.moe_feed_forward = SoftGatingMoE(
                dim=dim,
                num_experts=num_experts,
                num_classes=num_classes,
                top_k=top_k,
                hidden_size=hidden_size,
                moe_type=mode,
            )
        else:
            self.feed_forward = Mlp(
                in_features=dim, hidden_features=hidden_size, act_layer=act_layer
            )  # add hidden size
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask=None,
        tgt_mask=None,
        enc_output=None,
        task_categories=None,
        tgt_mask_id_bool=None,
    ):
        if self.mode == 'moe_attention':
            outputs = self.moe_attn(
                x,
                tgt_mask,
                task_categories,
                tgt_mask_id_bool,
            )
        else:
            attn_output = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, context=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        if self.mode == 'moe_ffn':
            outputs = self.moe_feed_forward(
                x,
                tgt_mask,
                task_categories,
                tgt_mask_id_bool,
            )
            ff_output = outputs[0]
        else:
            ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        if self.mode == 'none':
            return x
        else:
            return x, outputs[1], outputs[2], outputs[3]



class Geneformerwrapper(nn.Module):
    def __init__(
        self,
        model_path='ctheodoris/Geneformer',
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
        d_model: int = 512,
        nhead: int = 4,
        nlayers: int = 6,
        dropout: float = 0.02,
        d_ff: int = 512,
        position_embedding: Literal['sinusoidal', 'learnt'] = 'learnt',
    ):
        super().__init__()
        self.position_embedding = position_embedding
        if position_embedding == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
            )
        else:
            self.positional_encoding = LearntPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
            )
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_embedding = nn.Embedding(total_vocab_size, d_model, padding_idx=0)

        self.d_model = d_model

    #     self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.token_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        src: `torch.Tensor`
            shape ``[batch_size, seq_len, total_vocab_size]``
        src_mask: `torch.Tensor`
            shape ``[batch_size, seq_len]``

        Returns:
        --------
        output: `torch.Tensor`
            shape ``[batch_size, seq_len, total_vocab_size]``
        '''
        src_embedding = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(x=src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        if self.position_embedding == 'sinusoidal':
            src_embedding = self.positional_encoding(x=src_embedding)
        elif self.position_embedding == 'learnt':
            src_embedding = self.positional_encoding(x=src_embedding)
        return output

    

class CellGen(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int = 25426,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        d_ff: int = 2048,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        mlm_probability: float = 0.3,
        n_task_conditions: int = 3,
        encoder_type: Literal[
            'GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'
        ] = 'GF_frozen',
        moe_type: Literal['moe_attention', 'none', 'moe_ffn'] = 'none',
        position_embedding: Literal['sinusoidal', 'learnt'] = 'learnt',
        num_experts: int = 3,
        num_classes: int = 3,
        tokenid_to_genename_dict: Optional[dict] = None, # NEWLY ADDED
        num_perturbations: int = None,
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
        n_task_conditions: `int`
            Total number of conditions.
        encoder_type: `str`
            Mode of the encoder.
            Options: ['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder']
        moe_type: `str`
            Mode of the MoE.
            Options: ['moe_attention', 'none', 'moe_ffn']
        position_embedding: `str` (default: 'learnt')
            Positional encoding type: ['sinusoidal', 'learnt'].

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
            - 'expert_logits_list': Expert logits list from MoEs.
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
        self.num_perturbations = num_perturbations
        total_vocab_size = (
            tgt_vocab_size + 2
        )  # add one for each cls token
        self.mask_token = 1  # as defined in Geneformer
        self.token_embedding = nn.Embedding(total_vocab_size, d_model, padding_idx=0)
        
        
        self.position_embedding = position_embedding
        if position_embedding == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length + 2,  # Adjusted for perturbations
            )
        elif position_embedding == 'learnt':
            self.positional_encoding = LearntPositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length + 2,  # Adjusted for perturbations
            )
        else:
            raise ValueError(f'Invalid position embedding: {position_embedding}')
            
        if encoder_type in ['GF_frozen', 'GF_fine_tuned']:
            self.encoder_layers = Geneformerwrapper(mode=encoder_type)
        elif encoder_type == 'Transformer_encoder':
            self.encoder_layers = Encoder(
                total_vocab_size=total_vocab_size,
                max_seq_length=max_seq_length + 2, # Adjusted for perturbations
                d_model=d_model,
                position_embedding=position_embedding,
            )
        else:
            raise ValueError(f'Invalid encoder mode: {encoder_type}')
        self.encoder_type = encoder_type
        self.decoder_block = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    hidden_size=d_model,
                    dropout=dropout,
                    top_k=2,
                    num_experts=num_experts,
                    num_classes=num_classes,
                    mode=moe_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.moe_type = moe_type
        self.tokenid_to_genename_dict = tokenid_to_genename_dict # Store the tokenid_to_genename_dict
        
        if self.num_perturbations is not None:
            self.perturbation_embedding_layer = nn.Linear(
                self.num_perturbations, self.d_model
            )
        else:
            self.perturbation_embedding_layer = None


    def generate_mask(
        self, tgt_input_id, tgt_pad, mlm_probability=0.15, mask_mode='MASKGIT'
    ):
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
        mask_mode: `str`
            Masking mode: ['BERT', 'MASKGIT']
        Returns:
        --------
        tgt_input_id: `torch.Tensor`
            Target token input with masked tokens.
        labels: `torch.Tensor`
            True labels for masked tokens. Return -100 for non-masked tokens.
        '''
        device = tgt_input_id.device
        labels = tgt_input_id.clone()
        if mask_mode == 'BERT':
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
                2, self.tgt_vocab_size, labels.shape, dtype=torch.long, device=device
            )
            tgt_input_id[indices_random] = random_tokens[indices_random]
        elif mask_mode == 'MASKGIT':
            sample_length = torch.sum(~tgt_pad, dim=1)
            batch, seq_len = tgt_input_id.shape
            rand_time = uniform((batch,), device=device)
            rand_mask_probs = noise_schedule(rand_time, method='cosine')
            num_token_masked = (
                (torch.mul(sample_length, rand_mask_probs)).round().clamp(min=1)
            )
            # # exclude case where all tokens are masked based on sample length
            # # -2: avoid CLS token and leave at least one token unmasked
            # num_token_masked = num_token_masked.clamp(max=sample_length - 2)
            rand_int = torch.rand((batch, seq_len), device=device)
            # avoid masking pad and CLS token
            rand_int[tgt_pad] = 1
            # # mask CLS token by setting the last token to 0
            # # consequently, it will get highest rank after sorting at position 0
            # rand_int[:, seq_len - 1] = 0
            batch_randperm = rand_int.argsort(dim=-1)
            mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
            # do not mask CLS token
            mask[:, 0] = False
            tgt_input_id[mask] = self.mask_token
            labels[~mask] = -100
        return tgt_input_id, labels

    def call_encoder(self, src_input_id, src_attention_mask):
        if self.encoder_type in ['GF_frozen', 'GF_fine_tuned']:
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
        generate=False,
        labels=None,
        cls_positions=None,
        task_categories=None,
        tgt_mask_id_bool=None,
    ):
        for dec_layer in self.decoder_block:
            if self.moe_type == 'none':
                dec_embedding = dec_layer(
                    x=dec_embedding,
                    src_mask=src_attention_mask,
                    tgt_mask=tgt_pad,
                    enc_output=enc_output,
                )
                gate_logits = None
                expert_logits_list = None
                router_probs = None
            else:
                # see if concatenation of cls embedding
                (
                    dec_embedding,
                    gate_logits,
                    expert_logits_list,
                    router_probs,
                ) = dec_layer(
                    x=dec_embedding,
                    src_mask=src_attention_mask,
                    tgt_mask=tgt_pad,
                    enc_output=enc_output,
                    task_categories=task_categories,
                    tgt_mask_id_bool=tgt_mask_id_bool,
                )

        # Updated with CGPT 01
        outputs = {
            'cls_embedding': dec_embedding[:, 0, :],
            'mean_embedding': mean_nonpadding_embs(dec_embedding, tgt_pad),
        }
        
        decoder_logits = self.decoder_fc(dec_embedding)
        if labels is not None:
            outputs['dec_logits'] = decoder_logits
            outputs['labels'] = labels

        if generate is True:
            outputs['dec_logits'] = decoder_logits
            outputs['router_probs'] = router_probs
            outputs['dec_embedding'] = dec_embedding
        else:
            outputs['dec_embedding'] = dec_embedding
            outputs['mean_embedding'] = mean_nonpadding_embs(
                embs=dec_embedding,
                pad=tgt_pad,
            )
            outputs['cls_embedding'] = dec_embedding[:, 0, :]
            outputs['gate_logits'] = gate_logits
            outputs['expert_logits_list'] = expert_logits_list
            outputs['router_probs'] = router_probs
        if cls_positions is not None:
            outputs['cls_positions'] = cls_positions
        return outputs
    
    def gene_to_id(gene_list, gene_id_dictionary):
        '''
        Description:
        ------------
        Converting the list of perturbed genes into the GF ids.
        
        Parameters:
        -----------
        gene_list: 'list'
        List of genes perturbed for a condition
        
        gene_id_dictionary: honestly not sure what type this is
        Pickle file mapping genes with their GF ids
        
        Returns:
        --------
        outputs: `list`
            Outputting list of indices corresponding to the genes

        '''
        
        # reverse lookup of the id
        idx_for_gene = {v: k for k, v in gene_id_dictionary.items()}
        
        # get a list of all the ids corresponding to the genes in gene_list
        idx_list = [idx_for_gene.get(gene_name) for gene_name in gene_list]
        
        return idx_list
    
    
    
#     #### THIS FUNCTION IS FROM THE MAIN BRANCH ####
#     def concatenate_context(
#         self,
#         context_embedding_dict,
#         context_pad_dict,
#     ):
#         context_embedding_dict_ = context_embedding_dict.copy()
#         context_pad_dict_ = context_pad_dict.copy()
#         # if generate:
#         #     context_pad_dict_.pop(f'tgt_pad_t{tgt_time_step}')
#         context_tensors = list(context_embedding_dict_.values())
#         context_embedding = torch.cat(context_tensors, dim=1)
#         context_pads = list(context_pad_dict_.values())
#         context_pad = torch.cat(context_pads, dim=1)

#         return context_embedding, context_pad



    def forward(
        self,
        src_input_id: torch.Tensor,
        apply_attn_mask: Optional[bool] = False,
        task_categories: Optional[list] = None,
        tgt_input_id: Optional[torch.Tensor] = None,
        cls_positions: Optional[torch.Tensor] = None,
        generate_id: Optional[dict] = None,
        generate_pad: Optional[dict] = None,
        perturbation: Optional[list] = None, # PERTURBATION PASSED HERE
        nperts: Optional[int] = None, # PERTURBATION NUMBER PASSED HERE
        generate: Optional[bool] = False,
        perturbation_vectors: Optional[torch.Tensor] = None,
    ):
        '''
        Description:
        ------------
        Forward pass for the Seq2Seq model.

        Parameters:
        -----------
        src_input_id: `torch.Tensor`
            Source src token ids input.
        tgt_input_id: `Optional[dict]`
            Dictionary of target token ids input.
        generate_id_dict: `dict`
            Dictionary of target token inputs for generation.
        generate_pad_dict: `dict`
            Dictionary of target padding masks for generation.
        tgt_time_step: `int`
            Target time step.
        cls_positions: `torch.Tensor`
            CLS token positions.
        apply_attn_mask: `bool`
            Whether to mask tokens. Should not be masked for testing and generation.

        Returns:
        --------
        outputs: `dict`
            Output dictionary
        '''
        
        print(f"CELLGEN FWD tgt_input_id: {tgt_input_id}")
        
        src_attention_mask = generate_padding(src_input_id)
        # tgt_attention_mask = generate_padding(tgt_input_id)
        

        # only extract context for all the ones before the selected time step
        # rest will be padded
        tgt_embedding = self.token_embedding(tgt_input_id)
        
        print(f'DEBUGGING shape of tgt_embedding: {tgt_embedding.shape}')
        
        if self.position_embedding == 'sinusoidal':
            tgt_embedding = self.positional_encoding(tgt_embedding)
        elif self.position_embedding == 'learnt':
            tgt_embedding = self.positional_encoding(tgt_embedding)


            # Call the encoder to get encoder outputs
        enc_output = self.call_encoder(src_input_id, src_attention_mask)

        # Process perturbation_vectors
        if perturbation_vectors is not None and self.perturbation_embedding_layer is not None:
            # Embed perturbation_vectors
            perturbation_embeddings = self.perturbation_embedding_layer(perturbation_vectors)
            # Shape: (batch_size, d_model)

            # Expand to match encoder output dimensions
            perturbation_embeddings = perturbation_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, d_model)

            # Add positional encoding if necessary
            if self.position_embedding == 'sinusoidal':
                perturbation_embeddings = self.positional_encoding(perturbation_embeddings)
            elif self.position_embedding == 'learnt':
                perturbation_embeddings = self.positional_encoding(perturbation_embeddings)

            # Concatenate perturbation embeddings with encoder outputs
            combined_enc_output = torch.cat([enc_output, perturbation_embeddings], dim=1)

            # Update the attention mask to include perturbations
            perturbation_mask = torch.zeros(
                (enc_output.size(0), 1), dtype=torch.bool, device=enc_output.device
            )
            combined_src_attention_mask = torch.cat(
                [src_attention_mask, perturbation_mask], dim=1
            )
        else:
            combined_enc_output = enc_output
            combined_src_attention_mask = src_attention_mask

        # Proceed with decoding using combined_enc_output
        outputs = self.call_decoder(
            enc_output=combined_enc_output,
            src_attention_mask=combined_src_attention_mask,
            dec_embedding=tgt_embedding,
            tgt_pad=tgt_pad,
            generate=generate,
            cls_positions=cls_positions,
            task_categories=task_categories,
        )
        return outputs


class CountHead(nn.Module):
    def __init__(
        self,
        loss_mode: str = 'zinb',
        n_genes: int = 1424,  # Add n_genes parameter
        tgt_vocab_size: int = 25426,
        d_model: int = 512,
        dropout: float = 0.0,
    ):
        '''
        Description:
        ------------
        Count prediction head for the Seq2Seq model.

        Parameters:
        -----------
        loss_mode: `str`
            Loss mode. Options: ['mse', 'zinb', 'nb']
        tgt_vocab_size: `int`
            Target vocabulary size.
        d_model: `int`
            Token embedding dimension.
        dropout: `float`
            Dropout rate for the MLP.

        Returns:
        --------
        count_outputs: `dict`
            Output dictionary containing the following keys:
            - 'count_lognorm': Log-normalized count prediction for MSE loss.
            - 'count_mean': Mean count prediction for ZINB and NB loss.
            - 'count_dropout': Dropout count prediction for ZINB loss.

        '''
        super(CountHead, self).__init__()
        self.loss_mode = loss_mode
        self.n_genes = n_genes
        
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=d_model,
            drop=dropout,
        )
        # n_genes = tgt_vocab_size
        if self.loss_mode == 'mse':
            self.relu_output = nn.Sequential(nn.Linear(d_model, self.n_genes), nn.ReLU())
        elif self.loss_mode == 'zinb':
            self.linear_output = nn.Linear(d_model, self.n_genes)
            self.softmax_output = nn.Sequential(
                nn.Linear(d_model, self.n_genes), nn.Softmax(dim=-1)
            )
        elif self.loss_mode == 'nb':
            self.softmax_output = nn.Sequential(
                nn.Linear(d_model, self.n_genes), nn.Softmax(dim=-1)
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
        n_genes: int = 1424,
        tgt_vocab_size: int = 25426,
        d_model: int = 512,
        add_mask_id: bool = True,
        dropout: float = 0.0,
        time_steps: list = [1, 2],
        n_task_conditions: int = 3,
    ):
        '''
        
        look into geneformer 2048 dict to see what/if the masked token is <mask>
        
        Description:
        ------------
        Loads complete Seq2Seq model with count prediction head.
        Weights from pretrained seq2seq model are loaded into the model.
        Use CLS or mean embeddings for count prediction.

        Parameters:
        -----------
        pretrained_model: `nn.Module`
            Pretrained Seq2Seq model.
        loss_mode: `str`
            Loss mode. Options: ['mse', 'zinb', 'nb']
        tgt_vocab_size: `int`
            Target vocabulary size.
        d_model: `int`
            Token embedding dimension.
        add_mask_id: `bool`
            Whether to add mask token.
        dropout: `float`
            Dropout rate for the MLP.

        Returns:
        --------
        count_outputs: `dict`
            Output dictionary containing the following keys:
            - 'count_output_t{t}': Count prediction for time step t.
            - 'count_log_norm': Log-normalized count prediction for MSE loss.
            - 'count_mean': Mean count prediction for ZINB and NB loss.
            - 'count_dropout': Dropout count prediction for ZINB loss.
        '''
        super(CountDecoder, self).__init__()
        self.pretrained_model = pretrained_model
        self.embed_dim = d_model

        self.loss_mode = loss_mode
        # exclude pad (-1) to get the number of genes
        self.count_decoder = CountHead(
            loss_mode=loss_mode,
            n_genes=n_genes,
            d_model=d_model,
            dropout=dropout,
        )
        total_vocab_size = tgt_vocab_size + n_task_conditions  # add one for cls token
        if add_mask_id:
            # self.mask_token = total_vocab_size
            self.mask_token = 1
            print(total_vocab_size)

        self.time_steps = time_steps
        self.n_task_conditions = list(range(1, n_task_conditions + 1))
        self.cls_embedding = None

    def forward(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id_dict: dict,
    ):
        outputs = self.pretrained_model(
            src_input_id=src_input_id,
            tgt_input_id=tgt_input_id_dict['tgt_input_id_t1'],
            apply_attn_mask=False,
        )

        cls_embedding = outputs['mean_embedding']  # Shape: [batch_size, d_model]
        count_outputs_tmp = self.count_decoder.forward(cls_embedding)  # Output shape: [batch_size, n_genes]

        count_outputs = {}
        for t in self.time_steps:
            count_outputs[f'count_output_t{t}'] = count_outputs_tmp

        return count_outputs

    def call_padding(self, src_input_id, tgt_input_id_dict, time_steps):
        tgt_pad_dict = {}
        tgt_pad_dict['src_pad'] = generate_padding(src_input_id)
        for time_step in time_steps:
            tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']
            tgt_pad_dict[f'tgt_pad_t{time_step}'] = generate_padding(tgt_input_id)
        return tgt_pad_dict
    
    
    ## FROM THE MAIN BRANCH
    def generate(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id: torch.Tensor,
        max_len: int,
        perturbed_embeddings: Optional[torch.Tensor] = None,
        can_remask_prev_masked: bool = False,
        tgt_input_id_dict=None,
        topk_filter_thres: float = 0.9,
        temperature: float = 2.0,
        iterations: int = 18,
        mask_scheduler: str = 'cosine',
    ):
        
        
        tgt_pad = generate_padding(src_input_id)
        ids = torch.full_like(tgt_input_id, self.mask_token, dtype=torch.long)
        print(f'tgt_input_id: {tgt_input_id}')
        # ids[:, 0] = tgt_input_id[:, 0]  # Keep CLS token

        scores = torch.zeros_like(tgt_input_id, dtype=torch.float)
        
        
        # taking care of padding tokens
        scores = scores.masked_fill(tgt_pad, -torch.finfo().max)
        ids = ids.masked_fill(tgt_pad, 0)
        
        
        
        # Generate sequences
        outputs, generated_ids = self.generate_sequence(
            generate_id=ids,
            generate_pad=tgt_pad,
            src_input_id=src_input_id,
            perturbed_embeddings=perturbed_embeddings,
            mask_scheduler=mask_scheduler,
            can_remask_prev_masked=can_remask_prev_masked,
            topk_filter_thres=topk_filter_thres,
            starting_temperature=temperature,
            iterations=iterations,
            scores=scores,
        )

        # Obtain embeddings and predict counts
        cls_embedding = mean_nonpadding_embs(
            embs=outputs['dec_embedding'],
            pad=tgt_pad,
        )
        count_outputs = self.count_decoder.forward(cls_embedding)
        
        count_outputs[f'count_output_t1'] = count_outputs
        count_outputs[f'cls_embedding_t1'] = cls_embedding
        
        return count_outputs

    
    def generate_sequence(
        self,
        generate_id: torch.Tensor,
        generate_pad: torch.Tensor,
        src_input_id: torch.Tensor,
        perturbed_embeddings: Optional[torch.Tensor],
        scores: torch.Tensor,
        mask_scheduler: str,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        starting_temperature: float = 2.0,
        iterations: int = 18,
        perturbation: Optional[list] = None,
    ):
        max_neg_value = -torch.finfo(scores.dtype).max
        # scores[:, 0] = max_neg_value  # Exclude CLS token from being masked

        tmp_ids = generate_id.clone()
        ids_to_keep = torch.zeros_like(tmp_ids, dtype=torch.long)

        for iteration, steps_until_x0 in zip(
            torch.linspace(0, 1, iterations),
            reversed(range(iterations)),
        ):
            rand_mask_prob = noise_schedule(
                ratio=iteration,
                total_tokens=tmp_ids.shape[1],
                method=mask_scheduler,
            )
            batch_size, _ = scores.shape
            unmasked = (scores != max_neg_value).sum(dim=1)
            num_tokens_to_mask = (unmasked.float() * rand_mask_prob).long()
            mask = torch.zeros_like(scores, dtype=torch.bool)
            indices_to_mask = torch.topk(
                scores, num_tokens_to_mask.max(), dim=-1
            ).indices

            for i in range(batch_size):
                mask[i, indices_to_mask[i, : num_tokens_to_mask[i]]] = True
            tmp_ids = tmp_ids.masked_fill(mask, self.mask_token)

            ids_to_keep = torch.where(
                mask,
                torch.tensor(0, dtype=tmp_ids.dtype, device=tmp_ids.device),
                tmp_ids,
            )

            # Forward pass through the pretrained model
            outputs = self.pretrained_model(
                src_input_id=src_input_id,
                tgt_input_id=tmp_ids,
                perturbed_embeddings=perturbed_embeddings,
                perturbation = perturbation,
                generate_id=tmp_ids,
                generate_pad=generate_pad,
                apply_attn_mask=False,
                generate=True,
            )

            logits = outputs['dec_logits']  # Exclude CLS token
            
            # dec_log = outputs['dec_logits']
            
            # print(f'SHAPE OF outputs[dec_logits]: {dec_log.shape}')
            
            ## logits shape: batch * seq_length * n_tokens
            ## find the gene in the logits and then turn that to -inf
            
            

#             tmp_ids_ = tmp_ids[:, 1:].clone()
            
            
#             scores_ = scores[:, 1:].clone()
#             ids_to_keep_ = ids_to_keep[:, 1:].clone()
            
            
            # Removing id in question
            for sample in range(logits.shape[0]):
                unique_ids = torch.unique(ids_to_keep[sample])
                logits[sample, :, unique_ids] = -float('inf')
            filtered_logits = top_k(logits.clone(), topk_filter_thres)
            
            
            print(f'filtered_logits shape: {filtered_logits.shape}')
            
            
            temperature = starting_temperature * (
                steps_until_x0 / iteration
            )
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = tmp_ids == self.mask_token
            
            print(f'pred_ids shape: {pred_ids.shape}')
            print(f'tmp_ids_ shape: {tmp_ids.shape}')
            
            
            tmp_ids = torch.where(is_mask, pred_ids, tmp_ids)
            probs_without_temperature = logits.softmax(dim=-1)

            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None]).squeeze(-1)

            if not can_remask_prev_masked:
                scores = scores.masked_fill(~is_mask, max_neg_value)

            # scores[:, 1:] = scores
            # tmp_ids[:, 1:] = tmp_ids

        outputs['dec_embedding'] = outputs['dec_embedding']
        return outputs, tmp_ids


if __name__ == '__main__':
    # from T_perturb.Dataloaders.datamodule import GeneformerDataModule
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 400
    dropout = 0.1
    n_tokens = 200
    decoder = Block(
        dim=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        hidden_size=d_ff,
        dropout=dropout,
        context_dim=d_model,
    )
    transformer = CellGen(tgt_vocab_size=13)
    torch.manual_seed(42)
    src_input_id = torch.tensor(
        [
            [1, 2, 2, 5, 7, 6, 9, 8, 9, 6, 0, 0],
            [1, 3, 2, 4, 7, 4, 9, 3, 9, 6, 0, 0],
        ]
    )
    label_tensor = torch.tensor(
        [[1, 2, 2, 4, 5, 6, 9, 8, 9, 6, 0, 0], [1, 2, 2, 4, 5, 6, 9, 8, 9, 6, 0, 0]]
    )
    label_prob = torch.tensor(
        [
            [0.1, 0.25, 0.2, 0.4, 0.5, 0.6, 0.6, 0.8, 0.9, 1.0, 0.95, 0.92],
            [0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.6, 0.8, 0.9, 1.0, 0.95, 0.92],
        ]
    )
    # create logits with random probabilities adding up to 1 for each row
    # (B, seq_length, vocab_size)
    logits = torch.rand((2, 12, 10))
    logits = logits / logits.sum(dim=-1, keepdim=True)
    tgt_pad = torch.tensor(
        [
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
        ]
    )
    # generate probability matrix
    probability_matrix = (~tgt_pad).long()
    threshold = 0.5
    # TTransformer.select_unique_topk
    # (label_tensor, label_prob, tgt_pad, probability_matrix)
    transformer.eval()
    transformer.generate(
        src_input_id=src_input_id.to('cuda'),
        noise_schedule=noise_schedule,
        tgt_input_id=label_tensor.to('cuda'),
        tgt_vocab_size=10,
        seq_length=12,
    )
