"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math

import torch
from einops import rearrange
from torch import einsum, nn
from transformers import BertForMaskedLM

from T_perturb.Dataloaders.datamodule import GeneformerDataModule

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
#     """
#     Drop paths (Stochastic Depth) per sample
#     (when applied in main path of residual blocks).
#     """

#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


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

    def forward(self, x, context=None, tgt_mask=None, src_mask=None):
        h = self.heads
        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # b = batch size, n = sequence length
        # h = number of heads, d = dimension of each hea[d
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        max_neg_value = -torch.finfo(sim.dtype).max
        if tgt_mask is not None:
            # print(mask.shape)
            # mask = rearrange(mask, 'b ... -> b (...)')  # flattening mask
            sim = sim.masked_fill(tgt_mask == 0, max_neg_value)
            # print(mask.shape)
            # mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            # repeat mask for each head
        if src_mask is not None:
            sim = sim.masked_fill_(~src_mask[:, None, :], max_neg_value)
            # sim.masked_fill_(mask.to(device), max_neg_value)
        # else:
        #     raise ValueError('mask is None')

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # head
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


# class Block(nn.Module):q
#     def __init__(
#         self,q
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


class DecoderLayer(nn.Module):
    def __init__(
        self, dim, n_heads, d_head, hidden_size, dropout=0.0, context_dim=None
    ):
        super().__init__()
        self.self_attn = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.feed_forward = Mlp(
            in_features=dim, hidden_features=hidden_size
        )  # add hidden size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, tgt_mask=None, enc_output=None):
        attn_output = self.self_attn(x, tgt_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, context=enc_output)  # change this back
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Geneformerwrapper(nn.Module):
    def __init__(
        self,
        model_path='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/generative_modelling_omic/Geneformer',
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

        return embs


class TTransformer(nn.Module):
    def __init__(
        self,
        tgt_vocab_size=25426,
        d_model=256,
        num_heads=8,
        num_layers=1,
        d_ff=2048,
        max_seq_length=2048,
        dropout=0.0,
        mlm_probability=0.15,
    ):
        super(TTransformer, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.num_features = self.embed_dim = d_model
        self.mlm_probability = mlm_probability
        # initialize tensor for CLS token
        self.cls_token = torch.tensor(
            [tgt_vocab_size], dtype=torch.long
        )  # start at 25426, because of 0 Python indexing
        self.decoder_embedding = nn.Embedding(
            tgt_vocab_size + 1, d_model, padding_idx=0
        )
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = Geneformerwrapper()
        self.encoder_layers.eval()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_pad(self, tgt):
        tgt_pad = torch.tensor((tgt != 0), dtype=bool)
        return tgt_pad

    def create_extended_attention_mask_for_decoder(
        self, input_id, attention_mask, device=None
    ):
        # if device is not None:
        #     warnings.warn(
        #         "The `device` argument is deprecated and
        #           will be removed in v5 of Transformers.", FutureWarning
        #     )
        # else:
        #     device = attention_mask.device
        batch_size, seq_length = input_id.shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = (
            seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
            <= seq_ids[None, :, None]
        )
        # in case past_key_values are used we need to
        # add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, seq_length, prefix_seq_len),
                        device=device,
                        dtype=causal_mask.dtype,
                    ),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        return extended_attention_mask

    def generate_mask(self, src_attention_mask, tgt, tgt_pad):
        src_mask = src_attention_mask.unsqueeze(1).unsqueeze(2)
        # # repeat src mask
        # src_mask = src_mask.repeat(1, 1, tgt.size(1), 1)
        tgt_pad = tgt_pad.unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        nopeak_mask = nopeak_mask.to(self.device)
        tgt_mask = tgt_pad & nopeak_mask
        return src_mask, tgt_mask

    def prepare_tokens(self, x, mask=None):
        # B, nc, d = x.shape
        x = x + self.positional_encoding(
            x
        )  # add positional encoding to eagit fetch origin
        return x

    def forward(self, src_input_id, tgt_input_id):
        device = tgt_input_id.device
        tgt_input_id = torch.cat(
            (self.cls_token.expand(tgt_input_id.shape[0], -1).to(device), tgt_input_id),
            dim=1,
        )
        src_attention_mask = src_input_id != 0
        tgt_pad = self.generate_pad(tgt_input_id)
        if self.training:
            # _, tgt_mask = self.generate_mask(
            #     src_attention_mask, tgt_input_id, tgt_pad
            # )
            tgt_mask = self.create_extended_attention_mask_for_decoder(
                tgt_input_id, tgt_pad, device
            )
        else:
            tgt_mask, _ = (tgt_pad, None)
        src_embedded = self.encoder_layers(src_input_id, src_attention_mask)
        src_embedded[~src_attention_mask] = 0
        tgt_embedded = self.prepare_tokens(
            self.decoder_embedding(tgt_input_id), tgt_mask
        )
        enc_output = src_embedded
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, src_attention_mask, tgt_mask, enc_output)

        output = self.fc(dec_output)
        # mask cls
        tgt_input_id_ = tgt_input_id.clone()
        tgt_input_id_[:, 0] = 0
        if self.training:
            return output, tgt_input_id_
        else:
            return output, dec_output

    def generate(self, input_ids):
        # Assuming input_ids is a tensor with shape [batch_size, sequence_length]
        with torch.no_grad():
            attention_mask = input_ids != 0
            # Perform the necessary processing steps
            # (e.g., embedding, positional encoding)
            embedded_input = self.decoder_embedding(input_ids)
            embedded_input[~attention_mask] = 0
            embedded_input = self.prepare_tokens(embedded_input)
            enc_output = self.encoder_layers(input_ids, attention_mask=None)
            # Initialize the generated sequence with the <CLS> token
            generated_sequence = self.cls_token.expand(input_ids.size(0), 1, -1)
            # Generate the sequence step by step
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(
                    generated_sequence, attention_mask=None, enc_output=enc_output
                )
                logits = self.fc(
                    dec_output[:, -1, :]
                )  # Take the logits for the last position
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
        return generated_sequence


# def generate(
#         self, idx,
#         max_new_tokens, temperature=1.0,
#         do_sample=False, top_k=None
#         ):
#     """
#     Take a conditioning sequence of indices idx
#     (LongTensor of shape (b,t)) and complete
#     the sequence max_new_tokens times, feeding the
#     predictions back into the model each time.
#     Most likely you'll want to make sure to be in model.eval()
#     mode of operation for this.
#     """
#     for _ in range(max_new_tokens):
#         # if the sequence context is growing too long we must crop it at block_size
#         idx_cond = (idx if idx.size(1) <= self.block_size
#                     else idx[:, -self.block_size:])
#         # forward the model to get the logits for the index in the sequence
#         logits, _ = self(idx_cond)
#         # pluck the logits at the final step and scale by desired temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop the logits to only the top k options
#         if top_k is not None:
#             v, _ = torch.topk(logits, top_k)
#             logits[logits < v[:, [-1]]] = -float('Inf')
#         # apply softmax to convert logits to (normalized) probabilities
#         probs = F.softmax(logits, dim=-1)
#         # either sample from the distribution or take the most likely element
#         if do_sample:
#             idx_next = torch.multinomial(probs, num_samples=1)
#         else:
#             _, idx_next = torch.topk(probs, k=1, dim=-1)
#         # append sampled index to the running sequence and continue
#         idx = torch.cat((idx, idx_next), dim=1)

#     return idx

if __name__ == '__main__':
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
    transformer = TTransformer()

    # test dataloader
    data_module = GeneformerDataModule(
        src_folder='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_0h.dataset',
        tgt_folder='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_16h.dataset',
        max_len=334,
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()
    # # iterate through batches
    train_iterator = iter(dataloader)
    batch = next(train_iterator)
    print(batch['src_input_ids'].shape)
    print(batch)

    # (batch_size, seq_length)
    # position = PositionalEncoding(d_model, max_seq_length)
    # print(position(tgt_data).shape)
    # print(decoder(tgt_data, enc_output=src_data).shape)
    out, label = transformer(batch['src_input_ids'], batch['tgt_input_ids'])
    print(out)

    # src_data = torch.randint(20000,(10, 500))
    # src_attn_mask = torch.ones((10, 500))
    # src_attn_mask[:, 200:] = 0
    # tgt_data = torch.randint(20000,(10, n_tokens))
    # #pad
    # tgt_data[:, 100:] = 0
    # out, label = transformer(src_data, src_attn_mask, tgt_data)
