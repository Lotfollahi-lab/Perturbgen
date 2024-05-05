'''
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math

import numpy as np
import torch
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm import tqdm
from transformers import BertForMaskedLM


# noise schedule
def noise_schedule(ratio, total_tokens, method, exponent=2.0):
    '''
    Noise schedule from Google MaskGIT paper
    URL: https://github.com/google-research/maskgit/blob/
    1db23594e1bd328ee78eadcd148a19281cd0f5b8/maskgit/libml/mask_schedule.py#L21
    Last accessed: 2024-03-23
    '''
    if method == 'uniform':
        mask_ratio = 1.0 - ratio
    elif 'pow' in method:
        mask_ratio = 1.0 - ratio**exponent
    elif method == 'cosine':
        mask_ratio = torch.cos(ratio * math.pi * 0.5)
    elif method == 'log':
        mask_ratio = -torch.log2(ratio) / torch.log2(total_tokens)
    elif method == 'exp':
        mask_ratio = 1 - torch.exp2(-torch.log2(total_tokens) * (1 - ratio))
    # Clamps mask into [epsilon, 1)
    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.0)
    return mask_ratio


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


# use mlp with out feature = 1 for count decoder
# predict mask token or on everything (whole sequence length)
# use MSE (log norm)
# use ZINB
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
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
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)  # projection head
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
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


# class Block(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )

#     def forward(self, x, return_attention=False):
#         y, attn = self.attn(self.norm1(x))
#         if return_attention:
#             return attn
#         x = x + self.drop_path(y)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


# class Gating(nn.Module):
#     def __init__(self, dim, n_heads, d_head, dropout):
#         super().__init__()
#         self.gates = nn.ModuleList(
#             [nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid()) for _ in range(3)]
#         )
#         self.attention_blocks = nn.ModuleList(
#             [
#                 CrossAttention(
#                     query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
#                 )
#                 for _ in range(3)
#             ]
#         )

#     def forward(self, x, disable_gates=None):
#         # disable_gates is a list of gate indices to be forced to zero
#         if disable_gates is None:
#             disable_gates = []

#         gated_outputs = []
#         for i, attention_block in enumerate(self.attention_blocks):
#             block_output = attention_block(x)
#             if i in disable_gates:
#                 gated_output = block_output * 0  # Force the gated output to be zero
#             else:
#                 gate = self.gates[i](block_output)
#                 gated_output = block_output * gate
#             gated_outputs.append(gated_output)

#         # Combine gated outputs; this part remains unchanged
#         combined_output = torch.stack(gated_outputs, dim=-1).sum(dim=-1)
#         return combined_output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        hidden_size,
        dropout=0.0,
        context_dim=None,
        # top_k=2,
        # num_experts=3,
        # num_classes=3,
    ):
        super().__init__()
        self.self_attn_1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.self_attn_2 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        # induce sparsity in the attention mechanism using MoE
        # self.top_k = top_k
        # Initialize experts
        # self.experts = nn.ModuleList(
        #     [
        #         CrossAttention(
        #             query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        #         )
        #         for _ in range(num_experts)
        #     ]
        # )
        # # learn gate weights one on batch and one on token level
        # self.cls_gating_layer = nn.Linear(dim, num_experts)
        # # self.token_gating_layer = nn.Linear(dim, num_experts)
        # self.classifier_fc = nn.Linear(dim, num_classes)

        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        # self.register_buffer('top_k_mask', torch.zeros(num_experts, dtype=torch.bool))
        self.feed_forward = Mlp(
            in_features=dim, hidden_features=hidden_size
        )  # add hidden size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, tgt_mask=None, enc_output=None, moe=False):
        attn_output = self.self_attn_1(x, mask=tgt_mask)
        # if moe:
        #     x = self.norm1(x + self.dropout(attn_output))
        #     # get cls token for gating
        #     cls_features = x[:, 0, :]  # (batch_size, seq_len, dim)
        #     aggregated_cls_features = cls_features.mean(dim=0).unsqueeze(
        #         0
        #     )  # Mean pooling, [1, d_model]
        #     cls_gate_logits = self.cls_gating_layer(
        #         aggregated_cls_features
        #     )  # [1, num_experts]
        #     cls_gate_logits = cls_gate_logits.squeeze(0)
        #     cls_gate_probs = F.softmax(cls_gate_logits, dim=-1)
        #     # print(cls_gate_probs)
        #     top_k_values, top_k_indices = torch.topk(cls_gate_probs, self.top_k)
        #     # Compute outputs for the selected top-k experts and weight them
        #     cls_logit_outputs = []
        #     moe_outputs = torch.zeros_like(
        #         x
        #     )  # Assuming encoded has shape [seq_length, batch_size, d_model]
        #     for i, index in enumerate(top_k_indices.squeeze(0)):
        #         # print("expert", index)
        #         expert_output = self.experts[index](x)
        #         weight = top_k_values.squeeze(0)[i]
        #         # print(weight)
        #         cls_expert_embedding = expert_output[:, 0, :]
        #         cls_expert_logit = self.classifier_fc(cls_expert_embedding)
        #         cls_logit_outputs.append(cls_expert_logit)
        #         moe_outputs += expert_output * weight
        # else:
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.self_attn_2(x, mask=tgt_mask)
        # cls_logit_outputs = None

        # self.top_k_mask.fill_(False)
        # self.top_k_mask.scatter_(0, top_k_indices, True)
        # # Filter to keep only the top-k experts active
        # gate_logits = self.token_gating_layer(x)
        # gated_logits_topk = gate_logits[
        #     :, :, self.top_k_mask
        # ]  # Apply gating mask, [seq_length, batch_size, top_k]
        # gate_probs = F.softmax(gated_logits_topk, dim=-1)
        # # only select top expert - hard gating for tokens
        # top_experts_per_token = gate_probs.argmax(dim=-1)  # [seq_length, batch_size]
        # top_experts_indices = top_k_indices.squeeze(0)[
        #     top_experts_per_token
        # ]  # Map back to actual expert indices
        # # Efficient selection of outputs from the top expert for each token
        # moe_outputs = torch.zeros_like(x)
        # for i, expert in enumerate(self.experts):
        #     print("Expert", i)
        #     mask = top_experts_indices == (i)
        #     # compute proportion of tokens assigned to each expert
        #     proportion = mask.sum() / (mask.size(0) * mask.size(1))
        #     print(proportion)
        #     if mask.any():
        #         expert_output = expert(x, mask=tgt_mask)
        #         moe_outputs += expert_output * mask.unsqueeze(-1).float()

        x = self.norm2(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, context=enc_output, mask=src_mask)
        x = self.norm3(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm4(x + self.dropout(ff_output))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, n_time_steps):
        # train time steps and interpolation timestep
        # TODO: separate timestep positional encoding
        # and positional encoding for the ranks
        super(SinusoidalPositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        total_seq_length = n_time_steps * max_seq_length
        pe = torch.zeros(total_seq_length, d_model)
        position = torch.arange(0, total_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, tgt_time_step=None):
        if tgt_time_step is not None:
            start_pos = (tgt_time_step - 1) * self.max_seq_length
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


class Geneformerwrapper(nn.Module):
    def __init__(
        self,
        model_path='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'generative_modelling_omic/Geneformer/',
        output_attentions=False,
        output_hidden_states=True,
    ):
        super(Geneformerwrapper, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(
            model_path,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, src_input_id, src_attention_mask):
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=src_input_id, attention_mask=src_attention_mask
            )
            embs = outputs.hidden_states[-1]

        # do we need to set this to eval ?

        return embs


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(min, max)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


# sampling helper
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class Petra(nn.Module):
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
    ):
        super(Petra, self).__init__()
        self.num_features = self.embed_dim = d_model
        self.mlm_probability = mlm_probability
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        total_vocab_size = tgt_vocab_size + total_time_steps  # add one for cls token
        self.time_steps = time_steps
        self.mask_token = total_vocab_size
        total_vocab_size = total_vocab_size + 1  # add one for padding token
        self.token_embedding = nn.Embedding(total_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,  # Specify the GPU device
            n_time_steps=total_time_steps,
        )
        self.encoder_layers = Geneformerwrapper()
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=d_model,
                    n_heads=num_heads,
                    d_head=d_ff,
                    hidden_size=d_model,
                    dropout=dropout,
                    # top_k=2,
                    # num_experts=len(time_steps),
                    # num_classes=len(time_steps),
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_fc = nn.Linear(d_model, tgt_vocab_size)  # Specify the GPU device)
        self.dropout = nn.Dropout(dropout)

    def generate_pad(self, tgt_pad):
        tgt_pad = tgt_pad == 0
        return tgt_pad

    # def generate_pad_testing(
    #     self, tgt_input_id_dict, tgt_pad, mlm_probability=0.15, time_step=2
    # ):
    #     tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']
    #     labels = tgt_input_id.clone()
    #     # Initialize probability_matrix tensor without specifying device
    #     probability_matrix = torch.full_like(
    #         tgt_pad, mlm_probability, dtype=torch.float
    #     )
    #     cls_tgt_pad = tgt_pad.clone()
    #     cls_tgt_pad[:, 0] = True
    #     probability_matrix = probability_matrix.masked_fill(
    #         cls_tgt_pad, 0
    #     )  # add CLS token to the tokens
    #     tgt_mask = torch.bernoulli(probability_matrix).bool()
    #     labels[~tgt_mask] = -100

    #     return labels, tgt_mask

    def generate_mask(self, tgt_input_id, tgt_pad, mlm_probability=0.15):
        labels = tgt_input_id.clone()
        probability_matrix = torch.full_like(
            tgt_pad, mlm_probability, dtype=torch.float
        )
        # add CLS token to the tokens
        cls_tgt_pad = tgt_pad.clone()
        cls_tgt_pad[:, 0] = True
        probability_matrix = probability_matrix.masked_fill(cls_tgt_pad, 0)
        tgt_mask = torch.bernoulli(probability_matrix).bool()
        labels[~tgt_mask] = -100

        return tgt_mask, labels

    # def forward_with_cond_scale(
    #     self,
    #     *args,
    #     cond_scale = 3.,
    #     return_embed = False,
    #     generate = False,
    #     **kwargs
    # ):
    # if cond_scale == 1:
    #     return self.forward(
    #         *args,
    #         return_embed = return_embed,
    #         cond_drop_prob = 0., **kwargs
    #         )

    # logits, embed = self.forward(
    #     *args,
    #     return_embed = True,
    #     cond_drop_prob = 0., **kwargs
    #     )

    #     null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

    #     scaled_logits = null_logits + (logits - null_logits) * cond_scale

    #     if return_embed:
    #         return scaled_logits, embed

    #     return scaled_logits
    def call_padding(
        self,
        tgt_input_id_dict,
        time_steps,
    ):
        tgt_pad_dict = {}
        for time_step in time_steps:
            tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']
            tgt_pad_dict[f'tgt_pad_t{time_step}'] = self.generate_pad(tgt_input_id)
        return tgt_pad_dict

    def generate_src_mask(self, src_input_id):
        src_attention_mask = src_input_id == 0
        return src_attention_mask

    def call_tgt_mask(
        self,
        tgt_input_id,
        tgt_pad,
    ):
        tgt_mask, labels = self.generate_mask(
            tgt_input_id,
            tgt_pad,
            self.mlm_probability,
        )
        tgt_input_id = tgt_input_id.masked_fill(tgt_mask, self.mask_token)

        return labels, tgt_input_id

    def call_encoder(self, src_input_id, src_attention_mask):
        enc_output = self.encoder_layers(src_input_id, src_attention_mask.int())
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
        for dec_layer in self.decoder_layers:
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
            # outputs['moe_logits'] = moe_logits
            outputs['labels'] = labels
            outputs['selected_time_step'] = time_random
        if generate is True:
            outputs['dec_embedding'] = dec_embedding
            outputs['dec_logits'] = decoder_logits[:, 1:, :]
        else:
            outputs['dec_embedding'] = dec_embedding
        if cls_positions is not None:
            outputs['cls_positions'] = cls_positions
        return outputs

    def context_padding(self, src_attention_mask, tgt_pad_dict, context_time_steps):
        context_pad_list = []
        if len(context_time_steps) == 0:
            full_context_pad = src_attention_mask
        else:
            for time_step in context_time_steps:
                context_pad = tgt_pad_dict[f'tgt_pad_t{time_step}']
                context_pad_list.append(context_pad)
            context_pad_ = torch.cat(context_pad_list, dim=1)
            full_context_pad = torch.cat([src_attention_mask, context_pad_], dim=1)

        return full_context_pad

    def generate_context(
        self,
        enc_output,
        src_attention_mask,
        all_time_steps,
        tgt_time_step,
        tgt_input_id_dict,
        tgt_pad_dict,
        cls_positions=None,
        generate=False,
    ):
        # only retrieve context for the ones before the selected time step
        # rest will be padded
        rest_time_steps = [
            time_step for time_step in all_time_steps if time_step != tgt_time_step
        ]
        if len(rest_time_steps) == 0:
            full_context_embedding = enc_output
        else:
            context_embedding_list = []
            # retrieve the embeddings to provide as context
            # pad the rest of the time steps
            for time_step in rest_time_steps:
                # print("contex time step",time_step)
                # select all the ones before the selected time step
                context_time_steps = [
                    tmp_time_step
                    for tmp_time_step in rest_time_steps
                    if tmp_time_step < time_step
                ]

                context_pad = self.context_padding(
                    src_attention_mask=src_attention_mask,
                    tgt_pad_dict=tgt_pad_dict,
                    context_time_steps=context_time_steps,
                )
                if len(context_embedding_list) == 0:
                    context = enc_output
                else:
                    dec_embeddings = torch.cat(context_embedding_list, dim=1)
                    context = torch.cat([enc_output, dec_embeddings], dim=1)
                tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{time_step}']

                if generate:
                    # tgt input id already padded
                    tgt_pad = tgt_pad_dict[f'tgt_pad_t{time_step}']
                else:
                    tgt_pad = tgt_pad_dict[f'tgt_pad_t{time_step}']
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
                    context_embedding_list.append(dec_outputs['dec_embedding'])
            context_embeddings = torch.cat(context_embedding_list, dim=1)
            full_context_embedding = torch.cat([enc_output, context_embeddings], dim=1)

        return full_context_embedding

    def context_backprop(
        self,
        context_embedding,
        src_attention_mask,
        all_time_steps,
        tgt_time_step,
        tgt_input_id_dict,
        tgt_pad_dict,
        cls_positions=None,
        generate=False,
        not_masked=False,
    ):
        context_time_steps = [
            tmp_time_step
            for tmp_time_step in all_time_steps
            if tmp_time_step != tgt_time_step
        ]
        context_pad = self.context_padding(
            src_attention_mask=src_attention_mask,
            tgt_pad_dict=tgt_pad_dict,
            context_time_steps=context_time_steps,
        )

        selected_tgt_pad = tgt_pad_dict[f'tgt_pad_t{tgt_time_step}']
        selected_tgt_input_id = tgt_input_id_dict[f'tgt_input_id_t{tgt_time_step}']
        # only create maskings for the selected time step
        if not_masked or generate:
            labels = None
            # do not mask for embeddings for testing
            masked_tgt_input_id = selected_tgt_input_id
        else:
            labels, masked_tgt_input_id = self.call_tgt_mask(
                selected_tgt_input_id, selected_tgt_pad
            )
        selected_tgt_embedding = self.token_embedding(masked_tgt_input_id)
        selected_dec_embedding = self.positional_encoding(
            selected_tgt_embedding, tgt_time_step
        )
        outputs = self.call_decoder(
            enc_output=context_embedding,
            src_attention_mask=context_pad,
            dec_embedding=selected_dec_embedding,
            tgt_pad=selected_tgt_pad,
            time_random=tgt_time_step,
            generate=generate,
            labels=labels,
            cls_positions=cls_positions,
        )
        return outputs

    def forward(
        self,
        src_input_id,
        original_lens,
        tgt_input_id_dict=None,
        generate_id_dict=None,
        generate_pad_dict=None,
        tgt_time_step=None,
        cls_positions=None,
        not_masked=False,
    ):
        if tgt_input_id_dict:
            tgt_pad_dict = self.call_padding(
                tgt_input_id_dict,
                self.time_steps,
            )
        else:
            tgt_pad_dict = generate_pad_dict

        src_attention_mask = self.generate_src_mask(src_input_id)
        enc_output = self.call_encoder(src_input_id, src_attention_mask)
        # distinction between selected time step and rest time steps
        if not_masked:
            dec_embedding_list = []
            for time_step in self.time_steps:
                # need to retrieve embeddings for each of selected time step
                tgt_time_step = time_step
                # contex should be all the ones before the selected time step
                context_embedding = self.generate_context(
                    enc_output=enc_output,
                    src_attention_mask=src_attention_mask,
                    all_time_steps=self.time_steps,
                    tgt_time_step=tgt_time_step,
                    tgt_input_id_dict=tgt_input_id_dict,
                    tgt_pad_dict=tgt_pad_dict,
                    cls_positions=cls_positions,
                )
                outputs = self.context_backprop(
                    context_embedding=context_embedding,
                    src_attention_mask=src_attention_mask,
                    all_time_steps=self.time_steps,
                    tgt_time_step=tgt_time_step,
                    tgt_input_id_dict=tgt_input_id_dict,
                    tgt_pad_dict=tgt_pad_dict,
                    cls_positions=cls_positions,
                    not_masked=not_masked,
                )
                dec_embedding_list.append(outputs['dec_embedding'])
            outputs['dec_embedding'] = torch.cat(dec_embedding_list, dim=1)

        else:
            if tgt_time_step is None:
                tgt_time_step = np.random.choice(self.time_steps)
                tgt_input_id_dict = tgt_input_id_dict
                generate = False
            else:
                tgt_time_step = tgt_time_step
                tgt_input_id_dict = generate_id_dict
                generate = True
            # only extract context for all the ones before the selected time step
            # rest will be padded
            # ---Initialise the decoder embeddings
            # to provide as context for selected time step---
            context_embedding = self.generate_context(
                enc_output=enc_output,
                src_attention_mask=src_attention_mask,
                all_time_steps=self.time_steps,
                tgt_time_step=tgt_time_step,
                tgt_input_id_dict=tgt_input_id_dict,
                tgt_pad_dict=tgt_pad_dict,
                cls_positions=cls_positions,
                generate=generate,
            )
            outputs = self.context_backprop(
                context_embedding=context_embedding,
                src_attention_mask=src_attention_mask,
                all_time_steps=self.time_steps,
                tgt_time_step=tgt_time_step,
                tgt_input_id_dict=tgt_input_id_dict,
                tgt_pad_dict=tgt_pad_dict,
                cls_positions=cls_positions,
                generate=generate,
                not_masked=not_masked,
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
        tgt_vocab_size = tgt_vocab_size - 1
        # initialise multiple decoder for each time step
        self.count_decoder = CountHead(loss_mode, tgt_vocab_size, d_model, dropout)
        total_vocab_size = tgt_vocab_size + total_time_steps  # add one for cls token
        total_vocab_size = total_vocab_size  # add one for mask token
        if add_mask_id:
            self.mask_token = total_vocab_size

        self.time_steps = time_steps
        self.cls_embedding = None

    def generate_pad(self, tgt):
        tgt_ = tgt.clone().detach()
        tgt_pad = tgt_ == 0

        return tgt_pad

    def forward(
        self,
        src_input_id,
        tgt_input_id_dict,
        original_lens,
        cls_positions=None,
    ):
        count_outputs = {}
        # find length for a single time step
        outputs = self.pretrained_model.forward(
            src_input_id=src_input_id,
            tgt_input_id_dict=tgt_input_id_dict,
            original_lens=original_lens,
            not_masked=True,
        )

        count_outputs = {}
        for i, t in enumerate(self.time_steps):
            print(cls_positions)
            print(outputs['dec_embedding'].shape)
            cls_position = cls_positions[i]
            cls_embedding = outputs['dec_embedding'][:, cls_position, :]
            count_outputs_tmp = self.count_decoder.forward(cls_embedding)
            count_outputs[f'count_output_t{t}'] = count_outputs_tmp
        return count_outputs

    def generate(
        self,
        src_input_id,
        tgt_input_id_dict,
        original_lens,
        can_remask_prev_masked=False,
        topk_filter_thres=0.9,
        # time_steps=[1, 2, 3],
        # steps=18,
        temperature=2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        iterations=18,  # optimal iterations found in maskgit paper
        mask_scheduler='cosine',
        cls_positions=[0, 247, 494],
    ):
        starting_temperature = temperature
        demask_fn = self.pretrained_model
        generate_id_dict = {}
        generate_pad_dict = {}
        dec_embedding_list = []
        for i, time_step in enumerate(self.time_steps):
            tgt_input_id_key = f'tgt_input_id_t{time_step}'
            tgt_input_id = tgt_input_id_dict[tgt_input_id_key]
            tgt_pad = self.generate_pad(tgt_input_id)
            generate_pad_dict[f'tgt_pad_t{time_step}'] = tgt_pad
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
                original_lens=original_lens,
                demask_fn=demask_fn,
                mask_scheduler=mask_scheduler,
                can_remask_prev_masked=can_remask_prev_masked,
                topk_filter_thres=topk_filter_thres,
                starting_temperature=starting_temperature,
                iterations=iterations,
                scores=scores,
                tgt_time_step=time_step,
            )
            generate_id_dict[tgt_input_id_key] = generated_ids
            dec_embedding_list.append(outputs['dec_embedding'])
        outputs['dec_embedding'] = torch.cat(dec_embedding_list, dim=1)
        count_outputs = {}
        for i, t in enumerate(self.time_steps):
            cls_position = cls_positions[i]
            cls_embedding = outputs['dec_embedding'][:, cls_position, :]
            count_outputs_tmp = self.count_decoder.forward(cls_embedding)
            count_outputs[f'count_output_t{t}'] = count_outputs_tmp
            count_outputs[f'cls_embedding_t{t}'] = cls_embedding
        return count_outputs

    def generate_sequence(
        self,
        generate_id_dict,
        generate_pad_dict,
        src_input_id,
        original_lens,
        demask_fn,
        mask_scheduler,
        can_remask_prev_masked=False,
        topk_filter_thres=0.9,
        starting_temperature=2.0,
        iterations=18,
        scores=None,
        tgt_time_step=1,
    ):
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
                original_lens=original_lens,
                not_masked=False,
                tgt_time_step=tgt_time_step,
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
            # avoid predicting the same token
            scores_ = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores_ = rearrange(scores_, '... 1 -> ...')

            if not can_remask_prev_masked:
                dtype = scores_.dtype
                scores_ = scores_.masked_fill(~is_mask, -torch.finfo(dtype).max)
            # add cls token to the ids and update scores and ids
            scores[:, 1:] = scores_
            tmp_ids[:, 1:] = tmp_ids_

        return outputs, tmp_ids


if __name__ == '__main__':
    # from T_perturb.Dataloaders.datamodule import GeneformerDataModule
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 400
    dropout = 0.1
    n_tokens = 200
    decoder = DecoderLayer(
        dim=d_model,
        n_heads=num_heads,
        hidden_size=d_ff,
        dropout=dropout,
        d_head=64,
        context_dim=d_model,
    )
    transformer = Petra(tgt_vocab_size=13)
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
