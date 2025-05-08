'''
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
from typing import (
    Dict,
    List,
    Literal,
    Optional,
)

import numpy as np
import torch
from einops import rearrange, repeat
from scmaskgit.Modules.T_model import scmoscf
from torch import einsum, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers import BertForMaskedLM

from T_perturb.src.utils import (
    generate_pad,
    gumbel_sample,
    mean_nonpadding_embs,
    noise_schedule,
    top_k,
    uniform,
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


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        length: int,
        n_time_steps: int,
        encoder: Literal[
            'GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'
        ] = 'GF_frozen',
        mode: Literal[
            'time_pos_sin',
            'comb_sin',
            'sin_learnt',
            'time_pos_learnt',
        ] = 'time_pos_sin',
    ):
        '''
        Description:
        ------------
        Positional encoding for the transformer model.

        Parameters:
        -----------
        d_model: `int`
            Token embedding dimension.
        length: `int`
            Length of the positional encoding.
        n_time_steps: `int`
            Number of time steps for training.
        encoder: : Literal['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder']
            Transformer encoder type.
        mode: Literal['time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt']
            'time_pos_learn' cannot be used for time interpolation and extrapolation.
            -> the missing time step encoding is not available during training.
        Returns:
        --------
        x: `torch.Tensor`
            Token embeddings with positional encoding.
            Shape ``[batch_size, seq_len, d_model]``
        '''
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.length = length
        self.n_time_steps = n_time_steps
        self.encoder = encoder
        self.mode = mode

        if self.encoder == 'Transformer_encoder':
            # add one time step to included src time step
            n_time_steps = n_time_steps + 1
        elif self.encoder == 'scmaskgit':
            n_time_steps = n_time_steps + 2

        if self.mode == 'time_pos_sin':
            self.register_buffer(
                'time_pe', self._generate_sinusoidal_encoding(n_time_steps, d_model)
            )
            self.register_buffer(
                'pos_pe', self._generate_sinusoidal_encoding(length, d_model)
            )
        elif self.mode == 'comb_sin':
            total_seq_length = n_time_steps * length
            self.register_buffer(
                'pe', self._generate_sinusoidal_encoding(total_seq_length, d_model)
            )
        elif self.mode == 'sin_learnt':
            self.register_buffer(
                'time_pe', self._generate_sinusoidal_encoding(n_time_steps, d_model)
            )
            self.pos_encoding = self._generate_learnt_encoding(length, d_model)
            position_ids = torch.arange(length).expand((1, -1))
            self.register_buffer('pos_ids', position_ids)
        elif self.mode == 'time_pos_learnt':
            self.time_encoding = self._generate_learnt_encoding(n_time_steps, d_model)
            position_ids = torch.arange(n_time_steps).expand((1, -1))
            self.register_buffer('time_pos_ids', position_ids)
            self.pos_encoding = self._generate_learnt_encoding(length, d_model)
            position_ids = torch.arange(length).expand((1, -1))
            self.register_buffer('pos_ids', position_ids)
        else:
            raise ValueError(f'Invalid mode: {mode}')

    def _generate_sinusoidal_encoding(self, length, d_model, buffer_name='pe'):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(buffer_name, pe.unsqueeze(0))
        return pe.unsqueeze(0)

    def _generate_learnt_encoding(self, length, d_model):
        return nn.Embedding(length, d_model)

    def forward(self, x, tgt_time_step=None):
        if self.encoder in ['GF_frozen', 'GF_fine_tuned']:
            tgt_time_step_ = tgt_time_step - 1
        elif self.encoder == 'Transformer_encoder':
            # adjust for src embedding starting at 0
            tgt_time_step_ = tgt_time_step
        elif self.encoder == 'scmaskgit':
            # adjust for src embedding starting at 0
            tgt_time_step_ = tgt_time_step + 1
        if self.mode == 'time_pos_sin':
            time_pe = self.time_pe[:, tgt_time_step_]
            time_pe = time_pe.unsqueeze(0).expand(x.size(0), x.size(1), -1)
            pos_pe = self.pos_pe[:, : x.size(1)]
            return x + time_pe + pos_pe
        elif self.mode == 'comb_sin':
            start_pos = (tgt_time_step_) * self.length
            end_pos = start_pos + x.size(1)
            pe = self.pe[:, start_pos:end_pos]
            return x + pe
        elif self.mode == 'sin_learnt':
            time_pe = self.time_pe[:, tgt_time_step_]
            time_pe = time_pe.unsqueeze(0).expand(x.size(0), x.size(1), -1)
            pos_ids = self.pos_ids[:, : x.size(1)]
            pos_ids = pos_ids.expand(x.size(0), -1)
            pos_pe = self.pos_encoding(pos_ids)
            return x + time_pe + pos_pe
        elif self.mode == 'time_pos_learnt':
            time_pos_ids = self.time_pos_ids[:, tgt_time_step_]
            time_pos_ids = time_pos_ids.expand(x.size(0), -1)
            time_pe = self.time_encoding(time_pos_ids)
            pos_ids = self.pos_ids[:, : x.size(1)]
            pos_ids = pos_ids.expand(x.size(0), -1)
            pos_pe = self.pos_encoding(pos_ids)
            return x + time_pe + pos_pe


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        layer_norm: bool = False,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ln = nn.LayerNorm(hidden_features) if layer_norm else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln(x)
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
        return_attn: bool = False,
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
        self.return_attn = return_attn
        self.dim_head = dim_head
        self.num_heads = num_heads

    def normal_attention(self, q, k, v, h, mask=None, return_attn=False):
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
        if return_attn:
            return out, attn
        return out, attn

    def sdpa_attention(self, q, k, v, h, mask=None, return_attn=False, identity=None):
        _, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        if mask is not None:
            # Expand the mask to match the target shape:
            # [batch_size, num_heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, h, seq_len_q, seq_len_k)
            # negate mask so that padding tokens=False
            mask = ~mask
        if return_attn & (identity is not None):
            identity = identity.expand(q.size(0), self.num_heads, -1, -1)

        with sdpa_kernel(
            backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            out_ = scaled_dot_product_attention(
                query=q,
                key=k,
                value=identity if return_attn else v,
                attn_mask=mask,
                is_causal=False,
            )
        if return_attn:
            out = out_ @ v
            attn = out_
            # average attention weights over heads
            attn = attn.mean(dim=1)
        else:
            out = out_
            attn = None
        del out_
        out = rearrange(out, ' b h n d -> b n (h d)', h=h)
        return out, attn

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attention_mode='sdpa',
    ):
        h = self.num_heads
        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)
        if self.return_attn:
            identity = torch.eye(k.size(1), device=k.device)
        else:
            identity = None
        if attention_mode == 'normal':
            out, attn = self.normal_attention(
                q, k, v, h, mask, return_attn=self.return_attn
            )
        elif attention_mode == 'sdpa':
            out, attn = self.sdpa_attention(
                q,
                k,
                v,
                h,
                mask,
                return_attn=self.return_attn,
                identity=identity,
            )
        else:
            raise ValueError(f'Invalid attention mode: {attention_mode}')
        return self.to_out(out), attn


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
        return_attn: bool = False,
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
            query_dim=dim,
            num_heads=num_heads,
            dim_head=d_ff,
            dropout=dropout,
            return_attn=return_attn,
        )
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dim_head=d_ff,
            dropout=dropout,
            return_attn=return_attn,
        )
        self.norm3 = norm_layer(dim)
        self.feed_forward = Mlp(
            in_features=dim, hidden_features=hidden_size, act_layer=act_layer
        )  # add hidden size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, tgt_mask=None, enc_output=None):
        attn_out, self_attn_weigths = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out, cross_attn_weights = self.cross_attn(
            x, context=enc_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout(attn_out))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x, self_attn_weigths, cross_attn_weights


class Geneformerwrapper(nn.Module):
    def __init__(
        self,
        model_path='/lustre/scratch126/cellgen/team361/kl11/'
        't_generative/T_perturb/Geneformer/gf-12L-95M-i4096',
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
        if mode == 'GF_frozen':
            for param in self.model.parameters():
                param.requires_grad = False
        self.mode = mode
        self.model = self.model

    def forward(self, src_input_id, src_attention_mask):
        # reduce precision for memory efficiency
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


class scmaskgitwrapper(nn.Module):
    def __init__(
        self,
        # model_path='/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output1/checkpoints/20250107_1024_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=08.ckpt',
        model_path=(
            '/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/'
            'output2/checkpoints/20250110_2325_cellgen_train_masking_lr_5e'
            '-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=01.ckpt'
        ),
    ):
        '''
        Description:
        ------------
        Wrapper for scMaskGit model.

        Parameters:
        -----------
        model_path: `str`
            Path to the scMaskGit model.

        '''
        super(scmaskgitwrapper, self).__init__()
        if model_path is None:
            raise ValueError('Model path is required for scmaskgit encoder')

        self.model = scmoscf(
            # tgt_vocab_size=26717,
            tgt_vocab_size=20274,  # PBMC median
            d_model=768,
            num_heads=8,
            num_layers=6,
            d_ff=96,
            max_seq_length=4096,
            dropout=0.03,
        )
        pretrained_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        corrected_dict = {
            k.replace('transformer.', ''): v for k, v in pretrained_dict.items()
        }
        self.model.load_state_dict(corrected_dict)
        # self.model = self.model.to(torch.bfloat16)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, src_input_id):
        with torch.no_grad():
            outputs = self.model.forward(src_input_id=src_input_id)
        return outputs['dec_embedding']


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
    pos_encoding_mode: `str` (default: 'learnt')
        Positional encoding type: ['sinusoidal', 'learnt'].
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
        pad_token: int = 0,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'
        ] = 'time_pos_sin',
    ):
        super().__init__()
        self.pos_embedding = PositionalEncoding(
            d_model=d_model,
            length=max_seq_length,
            n_time_steps=n_time_steps,
            encoder='Transformer_encoder',
            mode=pos_encoding_mode,
        )
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            activation='gelu',
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_layers=nlayers,
            norm=nn.LayerNorm(d_model),
        )
        self.token_embedding = nn.Embedding(
            total_vocab_size, d_model, padding_idx=pad_token
        )
        nn.init.xavier_uniform_(self.token_embedding.weight)

        self.d_model = d_model
        self.total_vocab_size = total_vocab_size
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src_input_id: torch.Tensor, src_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        src_embedding = self.token_embedding(src_input_id) * math.sqrt(self.d_model)
        dec_embedding = self.pos_embedding(src_embedding, tgt_time_step=0)
        output = self.transformer_encoder(
            dec_embedding,
            src_key_padding_mask=src_attention_mask,
        )
        return output


class CytoMeister(nn.Module):
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
        pred_tps: List[int] = [
            1,
            2,
            3,
        ],
        n_total_tps: int = 3,
        mask_scheduler: str = 'cosine',
        encoder: Literal['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'] = (
            'GF_fine_tuned'
        ),
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'
        ] = 'time_pos_sin',
        return_attn: bool = False,
        pad_token: int = 0,
        context_mode: bool = True,
        context_tps: List[int] | None = None,
        encoder_path: str | None = None,
        condition_dict: Dict[str, Dict] | None = None,
        gene_to_rowid: Dict[str, int] | None = None,
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
        pre_tps: `list`
            List of time steps for training and testing.
            the proportion of tokens to mask.
        n_total_tps: `int`
            Total number of target time steps.
        mask_scheduler: `str`
            Masking scheduler defining
        mode: Literal['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder']
            Mode of the encoder.
        pos_encoding_mode:
            Literal['time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt']
            Positional encoding type.
        context_mode: `bool`
            Whether to use context mode, where other time steps are used as context.
        context_tps: `list`
            List of context time steps.

        Returns:
        --------
        outputs: `dict`
            Output dictionary containing the following keys:
            - 'dec_logits': Decoder logits.
            - 'labels': True labels for masked tokens.
            - 'selected_time_step': Selected time step.
            - 'dec_embedding': Decoder embeddings.
            - 'mean_embedding': Mean embeddings for non-padding tokens.
        '''
        super(CytoMeister, self).__init__()

        self.pos_embedding = PositionalEncoding(
            d_model=d_model,
            length=max_seq_length,
            n_time_steps=n_total_tps,
            encoder=encoder,
            mode=pos_encoding_mode,
        )
        self.num_features = self.embed_dim = d_model
        self.mlm_probability = mlm_probability
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.pred_tps = pred_tps
        self.context_tps = context_tps
        self.total_tps = list(range(1, n_total_tps + 1))
        self.mask_token = 1
        self.token_embedding = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=pad_token
        )
        if encoder in ['GF_frozen', 'GF_fine_tuned']:
            self.encoder_layers = Geneformerwrapper(mode=encoder)
        elif encoder == 'scmaskgit':
            print('-- Initializing scmaskgit model')
            self.encoder_layers = scmaskgitwrapper(encoder_path)
        elif encoder == 'Transformer_encoder':
            self.encoder_layers = Encoder(
                total_vocab_size=tgt_vocab_size,
                max_seq_length=max_seq_length,
                n_time_steps=n_total_tps,
                d_model=d_model,
                pos_encoding_mode=pos_encoding_mode,
                pad_token=pad_token,
            )
        else:
            raise ValueError(f'Invalid encoder mode: {encoder}')
        self.encoder = encoder

        self.decoder_block = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    hidden_size=d_model,
                    dropout=dropout,
                    return_attn=return_attn,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.pad_token = pad_token
        self.context_mode = context_mode
        self.mask_scheduler = mask_scheduler
        self.gene_to_rowid = gene_to_rowid
        self.condition_dict = condition_dict

    def generate_mask(
        self,
        tgt_input_id,
        tgt_pad,
        mask_mode='MASKGIT',
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
        if (mask_mode == 'BERT') and (self.mlm_probability is not None):
            # Masked language modeling for the target tokens.
            # Prepare masked tokens inputs/labels for masked language modeling:
            # 80% MASK, 10% random, 10% original.
            # Modified from Huggingface Transformers library:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L840 # noqa
            # Accessed: 2024-05-12
            probability_matrix = torch.full_like(
                tgt_pad, self.mlm_probability, dtype=torch.float
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
            rand_mask_probs = noise_schedule(
                rand_time,
                method=self.mask_scheduler,
                total_tokens=torch.tensor(seq_len, device=device),
            )
            num_token_masked = (
                (torch.mul(sample_length, rand_mask_probs)).round().clamp(min=1)
            )
            rand_int = torch.rand(batch, seq_len, device=device)
            # # exclude CLS token and set pad token to 1 to exclude from masking
            rand_int[tgt_pad] = 1
            batch_randperm = rand_int.argsort(dim=-1)
            mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
            tgt_input_id[mask] = self.mask_token
            labels[~mask] = -100
        return tgt_input_id, labels

    def call_padding(
        self,
        tgt_input_id_dict,
        time_steps,
    ):
        tgt_pad_dict = {}
        for time_step in time_steps:
            tgt_input_id = tgt_input_id_dict[f'tgt_input_ids_t{time_step}']
            tgt_pad_dict[f'tgt_pad_t{time_step}'] = generate_pad(tgt_input_id)
        return tgt_pad_dict

    def call_encoder(self, src_input_id, src_attention_mask):
        encoder_params = {'src_input_id': src_input_id}
        if self.encoder in ['GF_frozen', 'GF_fine_tuned']:
            # BERT mask: 1 for tokens to keep, 0 for tokens to mask. Thus, negate mask.
            src_attention_mask = ~src_attention_mask.clone().int()
        if self.encoder != 'scmaskgit':
            # scmaskgit model creates its own attention mask
            encoder_params['src_attention_mask'] = src_attention_mask
        enc_output = self.encoder_layers(**encoder_params)
        return enc_output

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
        decoder_embedding_l1 = []
        decoder_embedding_lmid = []
        decoder_l = 1
        for dec_layer in self.decoder_block:
            # see if concatenation of cls embedding
            dec_embedding, self_attn_weights, cross_attn_weights = dec_layer(
                x=dec_embedding,
                src_mask=src_attention_mask,
                tgt_mask=tgt_pad,
                enc_output=enc_output,
            )
            if decoder_l == 1:
                decoder_embedding_l1.append(dec_embedding)
            elif decoder_l == self.num_layers // 2:
                decoder_embedding_lmid.append(dec_embedding)

            if self_attn_weights is not None:
                self_attn_list.append(self_attn_weights)
            if cross_attn_weights is not None:
                cross_attn_list.append(cross_attn_weights)
            decoder_l += 1
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

    def generate_context(
        self,
        enc_output,
        src_attention_mask,
        tgt_time_step,
        all_time_steps,
        tgt_input_id_dict,
        tgt_pad_dict,
        cond_dict,
    ):
        context_embs_list = [enc_output]
        context_pad_list = [src_attention_mask]
        # retrieve the embeddings to provide as context
        # pad the rest of the time steps
        all_time_steps = sorted(all_time_steps)
        for time_step in all_time_steps:
            # exclude tgt_time_step from the context
            if time_step != tgt_time_step:
                context = torch.cat(context_embs_list, dim=1)
                # else:
                #     context = enc_output
                context_pad = torch.cat(context_pad_list, dim=1)
                tgt_input_id = tgt_input_id_dict[f'tgt_input_ids_t{time_step}']
                tgt_pad = tgt_pad_dict[f'tgt_pad_t{time_step}']

                with torch.no_grad():
                    # # ---classifier-free guidance---
                    tgt_embedding = self.token_embedding(tgt_input_id)
                    dec_embedding = self.pos_embedding(
                        tgt_embedding, tgt_time_step=time_step
                    )
                    # create context for the ones before the selected time step
                    # pad the rest
                    dec_outputs = self.call_decoder(
                        enc_output=context,
                        src_attention_mask=context_pad,
                        dec_embedding=dec_embedding,
                        tgt_pad=tgt_pad,
                        labels=None,
                        tgt_input_id=None,
                    )
                    context_embs_list.append(dec_outputs['dec_embedding'])
                    context_pad_list.append(tgt_pad)
        context_embedding = torch.cat(context_embs_list, dim=1)
        context_pad = torch.cat(context_pad_list, dim=1)
        return context_embedding, context_pad

    def forward(
        self,
        src_input_id: torch.Tensor,
        not_masked: bool = False,
        tgt_time_step: int | None = None,
        tgt_input_id_dict: dict | None = None,
        generate_id_dict: dict | None = None,
        generate_pad_dict: dict | None = None,
        cond_dict: torch.Tensor | None = None,
        **kwargs,
    ):
        '''
        Description:
        ------------
        Forward pass for the Seq2Seq model.
        Parameters:
        -----------
        src_input_id: `torch.Tensor`
            Source token input.
        not_masked: `bool`
            Whether to mask tokens. Should not be masked for testing and generation.
        tgt_time_step: `int`
            Target time step.
        tgt_input_id_dict: `Optional[dict]`
            Dictionary of target token inputs from different time steps.
        generate_id_dict: `Optional[dict]`
            Dictionary of target token inputs for generation.
        generate_pad_dict: `Optional[dict]`
            Dictionary of target padding masks for generation.
        Returns:
        --------
        outputs: `dict`
            Output dictionary
        '''
        if self.context_tps is None:
            all_modelling_tps = self.pred_tps
        else:
            all_modelling_tps = self.context_tps + self.pred_tps
        if tgt_input_id_dict:
            tgt_pad_dict = self.call_padding(
                tgt_input_id_dict,
                all_modelling_tps,
            )
        else:
            tgt_pad_dict = generate_pad_dict
        src_attention_mask = generate_pad(src_input_id)
        enc_output = self.call_encoder(src_input_id, src_attention_mask)
        if (not_masked) and (tgt_input_id_dict is not None):
            # not masked for count prediction and predicted embeddings
            sorted_time_steps = sorted(self.pred_tps)
            context_time_steps = (
                sorted(self.context_tps) if self.context_tps else sorted_time_steps
            )
        if not_masked is False:
            context_time_steps = (
                sorted(self.context_tps) if self.context_tps else sorted(self.pred_tps)
            )
            if tgt_time_step is None:
                # randomly select a time step for training
                sorted_time_steps = [np.random.choice(self.pred_tps)]
            elif generate_id_dict is not None:
                # MASKGIT generation
                tgt_input_id_dict = generate_id_dict
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
                    cond_dict=cond_dict,
                )
            if (not_masked is False) and (generate_id_dict is None):
                # apply masking during first stage of MLM training
                tgt_input_id, labels = self.generate_mask(
                    tgt_input_id,
                    tgt_pad,
                    mask_mode='MASKGIT',
                )
            else:
                # no true labels for MLM loss
                labels = None

            tgt_embedding = self.token_embedding(tgt_input_id)
            dec_embedding = self.pos_embedding(tgt_embedding, tgt_time_step)
            # does not include any context
            outputs = self.call_decoder(
                enc_output=context_output if self.context_mode else enc_output,
                src_attention_mask=context_mask
                if self.context_mode
                else src_attention_mask,
                dec_embedding=dec_embedding,
                tgt_pad=tgt_pad,
                labels=labels,
                tgt_input_id=tgt_input_id_dict[f'tgt_input_ids_t{tgt_time_step}'],
            )
            all_outputs[tgt_time_step] = outputs
        return all_outputs

    def generate(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id_dict: dict,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        temperature: float = 2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        iterations: int = 18,  # optimal of iterations in MaskGIT
        cond_length: int = 0,
        mask_scheduler: str = 'cosine',
        sequence_length: int | None = None,
        **kwargs,
    ):
        '''
        Description:
        ------------
        Generate sequences for the target tokens
        adopted from MaskGIT using the pretrained model.
        After T iterations when all tokens are predicted,
        use mean non-padding embeddings for count prediction.

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
        sequence_length: `int`
            Maximum length of the generated sequence.
        Returns:
        --------
        `tuple` containing:
            - count_outputs: `dict`
                Output dictionary containing the following keys:
                - 'count_output_t{t}': Count prediction for time step t.
                - 'cls_embedding_t{t}': CLS token embeddings for time step t.
            - 'generate_id_dict': Dictionary of generated token ids.
        '''
        generate_id_dict: Dict[str, torch.Tensor] = {}
        all_outputs: Dict[int, torch.Tensor] = {}
        if self.context_tps is not None:
            all_modelling_tps = self.pred_tps + self.context_tps
            all_modelling_tps = sorted(list(set(all_modelling_tps)))
        else:
            all_modelling_tps = sorted(self.pred_tps)
        tgt_pad_dict = self.call_padding(
            tgt_input_id_dict,
            all_modelling_tps,
        )
        # filter tgt_input_id_dict to include only all_modelling_tps
        tgt_input_id_dict = {
            k: v
            for k, v in tgt_input_id_dict.items()
            if int(k.split('_t')[-1]) in all_modelling_tps
        }
        for time_step in self.pred_tps:
            tgt_input_id_dict_ = {k: v.clone() for k, v in tgt_input_id_dict.items()}
            tgt_pad_dict_ = {k: v.clone() for k, v in tgt_pad_dict.items()}
            if sequence_length is not None:
                # use max shape instead of genes you like to generate
                pad_tensor = torch.ones_like(tgt_pad_dict_[f'tgt_pad_t{time_step}'])
                # pass sequence length for generation
                pad_tensor[:, sequence_length:] = 0
            else:
                # use sequence length from original input
                pad_tensor = tgt_pad_dict_[f'tgt_pad_t{time_step}']
            tgt_pad = generate_pad(pad_tensor)
            tgt_pad_dict_[f'tgt_pad_t{time_step}'] = tgt_pad
            tgt_input_id_key = f'tgt_input_ids_t{time_step}'
            tgt_input_id = tgt_input_id_dict[tgt_input_id_key]

            # create ids and scores matrix for each batch
            ids = torch.full_like(tgt_input_id, self.mask_token, dtype=torch.long)
            # add cls token to the ids
            if cond_length > 0:
                ids[:, :cond_length] = tgt_input_id[:, :cond_length]

            # replace the rest of the tokens with pad token
            ids[tgt_pad_dict[f'tgt_pad_t{time_step}']] = self.pad_token
            tgt_input_id_dict_[tgt_input_id_key] = ids
            # pad ids
            scores = torch.zeros_like(tgt_input_id, dtype=torch.float)
            outputs, generated_ids = self.generate_sequence(
                generate_id_dict=tgt_input_id_dict_,
                generate_pad_dict=tgt_pad_dict_,
                src_input_id=src_input_id,
                demask_fn=self,
                mask_scheduler=mask_scheduler,
                can_remask_prev_masked=can_remask_prev_masked,
                topk_filter_thres=topk_filter_thres,
                starting_temperature=temperature,
                iterations=iterations,
                scores=scores,
                tgt_time_step=time_step,
                cond_length=cond_length,
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
        cond_length: int = 0,
        generate_mode: Literal['topk', 'topk-margin'] = 'topk-margin',
        **kwargs,
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
        max_neg_value = -torch.finfo(scores.dtype).max
        scores[:, :cond_length] = max_neg_value
        tmp_ids = generate_id_dict[f'tgt_input_ids_t{tgt_time_step}'].clone()
        batch_size, seq_len = tmp_ids.shape
        # find total_tokens by find the numbers of 1s in the mask
        total_tokens = torch.sum(tmp_ids == 1, dim=1)
        ids_to_keep = torch.zeros_like(tmp_ids, dtype=torch.long)
        # # keep indices which are not masked except for the CLS token

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
            indices_to_mask = torch.topk(
                scores, num_tokens_to_mask.max(), dim=-1
            ).indices
            # Mask the top `num_tokens_to_mask` positions for each sample
            for i in range(batch_size):
                mask[i, indices_to_mask[i, : num_tokens_to_mask[i]]] = True
            tmp_ids.masked_fill_(mask, self.mask_token)
            # # keep indices which are not masked except for the CLS token
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
            logits = outputs[tgt_time_step]['dec_logits'][:, cond_length:, :]

            # exclude cls token
            tmp_ids_ = tmp_ids[:, cond_length:].clone()
            scores_ = scores[:, cond_length:].clone()
            ids_to_keep_ = ids_to_keep[:, cond_length:].clone()
            # Create a mask of already predicted tokens
            indices = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
            logits.scatter_(2, indices, max_neg_value)
            filtered_logits = top_k(logits.clone(), topk_filter_thres)
            temperature = starting_temperature * (
                steps_until_x0 / iterations
            )  # temperature is annealed
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = tmp_ids_ == self.mask_token
            tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)
            probs_without_temperature = logits.softmax(dim=-1)
            if generate_mode == 'topk-margin':
                # Get top-2 probabilities
                top2_probs, _ = torch.topk(probs_without_temperature, k=2, dim=-1)
                uncertainty_scores = torch.abs(
                    top2_probs[:, :, 0] - top2_probs[:, :, 1]
                )
                scores_ = 1 - uncertainty_scores
                scores_ = scores_.unsqueeze(-1)  # Now scores_ has shape [b, seq, 1]
            else:
                scores_ = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores_ = rearrange(scores_, '... 1 -> ...')

            if not can_remask_prev_masked:
                scores_.masked_fill_(~is_mask, max_neg_value)
            # add cls token to the ids and update scores and ids
            scores[:, cond_length:] = scores_
            tmp_ids[:, cond_length:] = tmp_ids_
        return outputs[tgt_time_step], tmp_ids


class CountHead(nn.Module):
    def __init__(
        self,
        loss_mode: str = 'zinb',
        n_genes: int = 25426,
        d_model: int = 256,  # 256
        dropout: float = 0.0,
        layer_norm: bool = False,  # args
    ):
        '''
        Description:
        ------------
        Count prediction head for the Seq2Seq model.
        Parameters:
        -----------
        loss_mode: `str`
            Loss mode. Options: ['mse', 'zinb', 'nb']
        n_genes: `int`
            number of genes to be predicted.
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
        self.input_dim = d_model  # Expecting concatenated input

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=d_model,
            drop=dropout,
            layer_norm=layer_norm,
        )
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
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f'Expected input with {self.input_dim} features, but got {x.shape[-1]}.'
            )
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
        d_model: int = 128,
        max_seq_length: int = 2048,
        d_condc: int | None = None,
        d_condt: int = 768,
        encoder: Literal['GF_frozen', 'GF_fine_tuned', 'Transformer_encoder'] = (
            'GF_fine_tuned'
        ),
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt'
        ] = 'time_pos_sin',
        layer_norm: bool = False,
        add_cell_time: bool = False,
        dropout: float = 0.0,
        pred_tps: list = [1, 2],
        n_total_tps: int = 3,
        n_genes: int = 25426,
        context_tps: list[int] | None = None,
        use_positional_encoding: bool = False,
    ):
        '''
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
        d_model: `int`
            Token embedding dimension.
        add_mask_id: `bool`
            Whether to add mask token.
        dropout: `float`
            Dropout rate for the MLP.
        pred_tps: `list`
            List of time steps for training and testing.
        context_tps: `list`
            List of context time steps.
        n_total_tps: `int`
            Total number of target time steps.
        n_genes: `int`
            Number of genes which counts should be regressed.
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
        self.add_cell_time = add_cell_time

        self.loss_mode = loss_mode
        if self.add_cell_time:
            self.use_positional_encoding = use_positional_encoding
            self.pos_embedding = (
                PositionalEncoding(
                    d_model=d_model,
                    length=max_seq_length,
                    n_time_steps=n_total_tps,
                    encoder=encoder,
                    mode=pos_encoding_mode,
                )
                if use_positional_encoding
                else None
            )

        if self.add_cell_time:
            if use_positional_encoding:
                d_condc = d_model

        if self.add_cell_time:
            input_size = d_model + d_condt
            if d_condc is not None:
                input_size += d_condc

            self.count_decoder = CountHead(loss_mode, n_genes, input_size, dropout)
        else:
            self.count_decoder = CountHead(loss_mode, n_genes, d_model, dropout)

        self.pred_tps = pred_tps
        self.context_tps = context_tps
        self.total_tps = list(range(1, n_total_tps + 1))
        self.cls_embedding = None
        if self.add_cell_time:
            self.condition_dict_oh = {}
            # Dynamically create entries for each key in 1..max_value
            for i in self.total_tps:
                # Create a list with i ones, then (max_value - i) zeros
                vector = [1.0] * i + [0.0] * (n_total_tps - i)
                self.condition_dict_oh[i] = torch.tensor(vector, dtype=torch.float32)
            self.condition_layer_time = Mlp(
                in_features=n_total_tps,
                hidden_features=int(d_condt / 2),
                out_features=d_condt,
                drop=dropout,
                layer_norm=layer_norm,
            )  # New MLP layer
            self.condition_layer_celltype: nn.Module | None = None
            if d_condc is not None:
                self.condition_layer_celltype = Mlp(
                    in_features=d_model,
                    hidden_features=d_condc * 2,
                    out_features=d_condc,
                    drop=dropout,
                    layer_norm=layer_norm,
                )

    def forward(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id_dict: dict,
    ):
        outputs = self.pretrained_model(
            src_input_id=src_input_id,
            tgt_input_id_dict=tgt_input_id_dict,
            not_masked=True,
        )
        count_outputs = {}
        for t in self.pred_tps:
            cls_embedding = outputs[t]['mean_embedding']
            if self.add_cell_time:
                if self.use_positional_encoding and self.pos_embedding is not None:
                    condition_emb_time = self.pos_embedding.time_pe[:, t + 1]
                else:
                    device = next(
                        self.parameters()
                    ).device  # Get the device of the model
                    condition_emb_time = self.condition_layer_time(
                        self.condition_dict_oh[t].to(device)
                    )
                    if self.condition_layer_celltype is not None:
                        condition_emb_celltype = self.condition_layer_celltype(
                            outputs[t]['dec_embedding'][:, 1, :]
                        )  # Use one-hot
                        condition_emb_time = condition_emb_time.unsqueeze(0).expand(
                            condition_emb_celltype.shape[0], -1
                        )
                        condition_emb = torch.cat(
                            (condition_emb_time, condition_emb_celltype), dim=1
                        )
                    else:
                        condition_emb = condition_emb_time
                cls_embedding = torch.cat((cls_embedding, condition_emb), dim=1)
            count_outputs_tmp = self.count_decoder.forward(cls_embedding)
            count_outputs[f'count_output_t{t}'] = count_outputs_tmp

        return count_outputs

    def generate_counts(
        self,
        src_input_id: torch.Tensor,
        tgt_input_id_dict: dict,
        can_remask_prev_masked: bool = False,
        topk_filter_thres: float = 0.9,
        temperature: float = 2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        iterations: int = 18,  # optimal of iterations in MaskGIT
        mask_scheduler: str = 'cosine',
        sequence_length: int = 2048,
        cond_length: int = 0,
    ):
        outputs, generate_id_dict = self.pretrained_model.generate(
            src_input_id=src_input_id,
            tgt_input_id_dict=tgt_input_id_dict,
            can_remask_prev_masked=can_remask_prev_masked,
            topk_filter_thres=topk_filter_thres,
            temperature=temperature,
            iterations=iterations,
            mask_scheduler=mask_scheduler,
            sequence_length=sequence_length,
            cond_length=cond_length,
        )

        count_outputs = {}
        for t in self.pred_tps:
            # cls_embedding = outputs['dec_embedding'][:, 0, :]
            cls_embedding = outputs[t]['mean_embedding']
            if self.add_cell_time:
                if self.use_positional_encoding and self.pos_embedding is not None:
                    condition_emb_time = self.pos_embedding.time_pe[:, t + 1].to(
                        outputs[t]['mean_embedding'].device
                    )
                else:
                    device = cls_embedding.device
                    condition_emb_time = self.condition_layer_time(
                        self.condition_dict_oh[t].to(device)
                    )
                    if self.condition_layer_celltype is not None:
                        condition_emb_celltype = self.condition_layer_celltype(
                            outputs[t]['dec_embedding'][:, 1, :]
                        )
                        condition_emb_time = condition_emb_time.unsqueeze(0).expand(
                            condition_emb_celltype.shape[0], -1
                        )
                        condition_emb = torch.cat(
                            (condition_emb_time, condition_emb_celltype), dim=1
                        )
                    else:
                        condition_emb = condition_emb_time
                cls_embedding = torch.cat((cls_embedding, condition_emb), dim=-1)
                count_outputs[f'count_output_t{t}'] = self.count_decoder.forward(
                    cls_embedding
                )
                count_outputs[f'cls_embedding_t{t}'] = outputs[t]['mean_embedding']
            else:
                # cls_embedding = outputs['dec_embedding'][:, 0, :]
                count_outputs[f'count_output_t{t}'] = self.count_decoder.forward(
                    outputs[t]['mean_embedding']
                )
                count_outputs[f'cls_embedding_t{t}'] = outputs[t]['mean_embedding']
        return count_outputs, generate_id_dict
