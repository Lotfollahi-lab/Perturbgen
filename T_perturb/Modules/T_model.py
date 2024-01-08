"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from torch import nn, einsum
from einops import rearrange, repeat
import torch
import torch.nn as nn
import math
from transformers import BertForMaskedLM


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
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
            print(mask.shape)
            print(sim.shape)
            sim.masked_fill_(mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, d_head, hidden_size, dropout=0., context_dim=None):
        super().__init__()

        self.self_attn = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.cross_attn = CrossAttention(query_dim=dim, context_dim=context_dim,
                                         heads=n_heads, dim_head=d_head, dropout=dropout)
        self.feed_forward = Mlp(in_features=dim, hidden_features=hidden_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, tgt_mask=None, enc_output=None):
        attn_output = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, context=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Geneformerwrapper(nn.Module):
    def __init__(self, model_path="/lustre/scratch126/cellgen/team205/ml19/Arian/Geneformer/geneformer-6L-30M/"
                 , output_attentions=False, output_hidden_states=True):
        super(Geneformerwrapper, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(model_path, output_attentions=output_attentions,
                                                     output_hidden_states=output_hidden_states).to("cuda")

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=x["input_ids"].to("cuda"),
                attention_mask=x["attention_mask"].to("cuda")
            )
            embs = outputs.hidden_states[-1]

        return embs


class TTransformer(nn.Module):
    def __init__(self, tgt_vocab_size=25000, d_model=256, num_heads=8, num_layers=1, d_ff=2048,
                 max_seq_length=2048, dropout=0.0, mlm_probability=0.15):
        super(TTransformer, self).__init__()
        self.num_features = self.embed_dim = d_model
        self.mlm_probability = mlm_probability
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_label = nn.Parameter(torch.tensor(False))
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = Geneformerwrapper()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        labels = tgt.clone()
        src_mask = torch.tensor((src != 0),dtype=bool)
        tgt_pad = torch.tensor((tgt != 0),dtype=bool)
        tgt_pad = torch.cat((self.cls_label.expand(tgt_pad.shape[0], 1), tgt_pad), dim=1)
        # seq_length = tgt.size(1)
        probability_matrix = torch.full(tgt_pad.shape, self.mlm_probability)
        probability_matrix.masked_fill(~tgt_pad, 0)
        tgt_mask = torch.bernoulli(probability_matrix).bool()

        # nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        # tgt_mask = tgt_mask & nopeak_mask

        labels[~tgt_mask] = -100
        # labels = torch.cat((self.cls_label.expand(labels.shape[0],1), labels), dim=1)

        return src_mask, tgt_mask, labels

    def prepare_tokens(self, x):
        B, nc, d = x.shape
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.positional_encoding(x)

        return x

    def forward(self, src, tgt):
        src_mask, tgt_mask, labels = self.generate_mask(src, tgt)

        # src_embedded = self.encoder_layers(src)
        src_embedded = src
        tgt_embedded = self.prepare_tokens(self.decoder_embedding(tgt))
        enc_output = src_embedded
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, src_mask, tgt_mask, enc_output)

        output = self.fc(dec_output)
        return output, labels


if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 400
    dropout = 0.1
    n_tokens = 200
    decoder = DecoderLayer(dim=d_model, n_heads=num_heads, hidden_size=d_ff, dropout=dropout, d_head=64,
                           context_dim=d_model)
    transformer = TTransformer()
    # Generate random sample data

    src_data = torch.randint(20000,(10, 500, d_model))
    tgt_data = torch.randint(20000,(10, n_tokens))
    # (batch_size, seq_length)
    position = PositionalEncoding(d_model, max_seq_length)
    # print(position(tgt_data).shape)
    # print(decoder(tgt_data, enc_output=src_data).shape)
    print(transformer(src_data, tgt_data).shape)