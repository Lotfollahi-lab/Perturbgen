'''
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
import torch
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm import tqdm
from transformers import BertForMaskedLM

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


class DecoderLayer(nn.Module):
    def __init__(
        self, dim, n_heads, d_head, hidden_size=None, dropout=0.0, context_dim=None
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

# To do: path not hardcoded
class Geneformerwrapper(nn.Module):
    def __init__(
        self,
        model_path='/lustre/groups/imm01/workspace/irene.bonafonte/Software/Geneformer/geneformer-12L-30M',
        output_attentions=False,
        output_hidden_states=True,
        tune=False,
    ):
        super(Geneformerwrapper, self).__init__()
        self.tune = tune
        self.model = BertForMaskedLM.from_pretrained(
            model_path,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if self.tune:
            for param in self.model.parameters():
                param.requires_grad = True            
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, src_input_id, src_attention_mask):
        if self.tune:
            outputs = self.model.forward(
                input_ids=src_input_id, attention_mask=src_attention_mask
            )
            embs = outputs.hidden_states[-1]
        else:
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=src_input_id, attention_mask=src_attention_mask
                )
                embs = outputs.hidden_states[-1]

        return embs


# noise schedule
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


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
        d_encoded_input=None,   
        d_perturbation_embed=None,     
        perturbation_modeling=None,
        base_path: str = '/lustre/groups/imm01/workspace/irene.bonafonte',
        tune_geneformer=False,
    ):
        super(Petra, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = self.embed_dim = d_model
        self.mlm_probability = mlm_probability
        self.perturbation_modeling = perturbation_modeling
        self.tgt_vocab_size = tgt_vocab_size

        self.register_buffer('cls_token', torch.tensor([tgt_vocab_size], dtype=torch.long)) # start at 25426, because of 0 Python indexing
        total_vocab_size = tgt_vocab_size + 1

        self.mask_token = total_vocab_size
        total_vocab_size = total_vocab_size + 1

        # self.masked_embed = nn.Parameter(torch.zeros(1, self.embed_dim))
        print('embedding size problem')
        self.token_embedding = nn.Embedding(
            total_vocab_size, d_model, padding_idx=0 #, device=self.device
        )
        print(self.token_embedding.weight.shape)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # self.positional_encoding = self.positional_encoding.to(self.device)
        self.encoder_layers = Geneformerwrapper(model_path=f'{base_path}/Software/Geneformer/geneformer-12L-30M', tune=tune_geneformer)
        # self.encoder_layers = self.encoder_layers.to(self.device)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dim=d_model, n_heads=num_heads, d_head=d_ff, dropout=dropout, context_dim=d_encoded_input) for _ in range(num_layers)]
        )
        # self.decoder_layers = self.decoder_layers.to(self.device)
        if self.perturbation_modeling is not None:
            if d_encoded_input != d_perturbation_embed:
                self.fc_pertReshape = nn.Linear(d_perturbation_embed, d_encoded_input)
            else:
                self.fc_pertReshape = None
    
        self.fc = nn.Linear(d_model, tgt_vocab_size) #, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def generate_pad(self, tgt):
        tgt_pad = tgt == 0
        return tgt_pad

    def generate_mask(self, tgt, tgt_pad, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Modified from Huggingface Transformers library:
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L840 # noqa
        Accessed: 2024-05-12
        """
        device = tgt.device
        labels = tgt.clone()
        probability_matrix = torch.full_like(
            tgt_pad, mlm_probability, dtype=torch.float # device=self.device
        )
        # cls_tgt_pad = (tgt == self.cls_token) | (tgt == self.perturbation_token) | (tgt == 0)
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
        tgt[indices_masked] = self.mask_token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
            & masked_indices
            & ~indices_masked
        )
        # +1 to exclude pad and cls tokens from random token selection
        random_tokens = torch.randint(
            1, self.tgt_vocab_size, labels.shape, dtype=torch.long, device=device
        )
        tgt[indices_random] = random_tokens[indices_random]
        # 10% remain unmasked
        return tgt, labels        

    def prepare_tokens(self, x):
        # B, nc, d = x.shape
        # add positional encoding to each token
        # x = x + self.positional_encoding(x)
        x = self.positional_encoding(x)
        return x

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

    def forward(
        self,
        src_input_id,
        tgt_input_id,
        perturbation_id,
        perturbation_embedding,
        original_lens,
        generate=False,
    ):
        src_attention_mask = src_input_id == 0
        # convert to numeric type
        src_attention_mask_int = (~src_attention_mask).int()
        src_embedded = self.encoder_layers(src_input_id, src_attention_mask_int) # 0 for padded

        # add embedding at the begining of the src_embedding for perturbed genes and update mask
        if self.perturbation_modeling is not None:
            # if size is different than d_model, reshape via linear layer:
            if self.fc_pertReshape is not None:
                perturbation_embedding = self.fc_pertReshape(perturbation_embedding)

            for i in range(src_input_id.shape[0]):
                # move sentence 2 positions to the right to leave place for the perturbation embeddings
                src_embedded[i,2:,:] = src_embedded[i,:-2,:].clone()
                src_attention_mask[i,2:] = src_attention_mask[i,:-2].clone()
            
                # add perturbation/s embeddings in the first positions
                src_embedded[i,:2,:] = perturbation_embedding[i]
                if len(perturbation_id[i]) == 1:
                    src_attention_mask[i,0] = False
                    src_attention_mask[i,1] = True
                else:
                    src_attention_mask[i,:2] = False

        # append cls token at the beginning of the target input ids
        tgt_input_id = torch.cat(
            (
                self.cls_token.expand(tgt_input_id.shape[0], -1),
                tgt_input_id,
            ),
            dim=1,
        )
        tgt_pad = self.generate_pad(tgt_input_id)

        # Mask tokens for learning
        if generate:
            labels = None
        else:
            tgt_input_id, labels = self.generate_mask(
                tgt_input_id, tgt_pad, self.mlm_probability
            )
        # tokens to embeddings
        tgt_embedded_mask = self.token_embedding(tgt_input_id)
        tgt_embedded_mask = self.prepare_tokens(tgt_embedded_mask) # add positional encoding

        enc_output = src_embedded
        dec_embedding = tgt_embedded_mask

        for dec_layer in self.decoder_layers:
            dec_embedding = dec_layer(
                dec_embedding, src_attention_mask, tgt_pad, enc_output
            )
        logits = self.fc(dec_embedding)

        outputs = {}
        if generate:
            outputs['cls_embedding'] = dec_embedding[:, 0, :]
            outputs['logits'] = logits[:, 1:, :]  # ignore CLS token
        else:
            outputs['logits'] = logits
            outputs['labels'] = labels
            outputs['dec_embedding'] = dec_embedding

        return outputs


class CountHead(nn.Module):
    def __init__(
        self,
        loss_mode: str = 'zinb',
        tgt_vocab_size: int = 25426,
        d_model: int = 256,
        dropout: float = 0.0,
        perturbation_modeling: str = None,
    ):
        super(CountHead, self).__init__()
        self.loss_mode = loss_mode
        self.perturbation_modeling = perturbation_modeling
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=d_model,
            drop=dropout,
        )
        n_genes = tgt_vocab_size - 1
        if self.loss_mode == 'mse':
            self.relu_output = nn.Sequential(nn.Linear(d_model, n_genes), nn.ReLU())
            self.zero_logit = nn.Sequential(nn.Linear(d_model, n_genes), nn.Sigmoid())

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
        count_outpus = {}
        mlp_output = self.mlp(x)
        mlp_output = nn.functional.normalize(mlp_output, dim=-1, p=2)
        if self.loss_mode == 'mse':
            count_outpus['count_lognorm'] = self.relu_output(mlp_output)
            count_outpus['zero_probs'] = self.zero_logit(mlp_output)

        elif self.loss_mode == 'zinb':
            count_outpus['count_mean'] = self.softmax_output(mlp_output)
            count_outpus['count_dropout'] = self.linear_output(mlp_output)
        elif self.loss_mode == 'nb':
            count_outpus['count_mean'] = self.softmax_output(mlp_output)
        return count_outpus


class CountDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module = None,
        loss_mode: str = 'zinb',
        tgt_vocab_size: int = 25426,
        d_model: int = 256,
        add_mask_id: bool = True,
        dropout: float = 0.0,
        perturbation_modeling=None,
        tune_pretrained=True,
    ):
        super(CountDecoder, self).__init__()
        self.pretrained_model = pretrained_model
        if not tune_pretrained:
            for _, param in self.pretrained_model.named_parameters():
                param.requires_grad = False
        self.embed_dim = d_model
        if add_mask_id:
            total_vocab_size = tgt_vocab_size + 1  # CLS and masked token
            self.mask_token = total_vocab_size
        self.loss_mode = loss_mode
        self.decoder = CountHead(loss_mode, tgt_vocab_size, d_model, dropout)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cls_embedding = None
        self.perturbation_modeling = perturbation_modeling

    def generate_pad(self, tgt):
        tgt_ = tgt.clone().detach()
        tgt_pad = tgt_ == 0

        return tgt_pad

    def forward(
        self,
        src_input_id,
        tgt_input_id,
        original_lens,
        perturbation_id,
        perturbation_embedding,
        generate=False,
    ):
        outputs = self.pretrained_model.forward(
            src_input_id=src_input_id,
            tgt_input_id=tgt_input_id,
            original_lens=original_lens,
            perturbation_id=perturbation_id,
            perturbation_embedding=perturbation_embedding,
            generate=generate,
        )
        cls_embedding = outputs['dec_embedding'][:, 0, :]

        # use cls token for count prediction
        count_outputs = self.decoder.forward(cls_embedding)

        return count_outputs

    def generate(
        self,
        src_input_id,
        noise_schedule,
        tgt_input_id,
        original_lens,
        perturbation_id,
        perturbation_embedding,
        can_remask_prev_masked=False,
        topk_filter_thres=0.9,
        temperature=2.0,  # keep in range 2.0-3.0
        # self_cond_prob=0.9,
        timesteps=18,  # optimal iterations found in maskgit paper
    ):
        tgt_pad = self.generate_pad(tgt_input_id)

        batch_size = tgt_input_id.shape[0]
        seq_len = tgt_input_id.shape[1]
        shape = (batch_size, seq_len)
        # create ids and scores matrix for each batch
        # exclude CLS token from token
        ids = torch.full(shape, self.mask_token, dtype=torch.long, device=tgt_input_id.device)

        # pad ids
        scores = torch.zeros(shape, dtype=torch.float, device=tgt_input_id.device)
        starting_temperature = temperature
        demask_fn = self.pretrained_model

        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=tgt_input_id.device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):
            # mask scheduler function, gamma
            rand_mask_prob = noise_schedule(timestep)
            # pad scores and ids
            scores = scores.masked_fill(tgt_pad, -torch.finfo().max)

            ids = ids.masked_fill(tgt_pad, 0)
            ids_to_keep = torch.zeros_like(ids, dtype=torch.long)

            for i, score in enumerate(scores):
                # count zeros in each row
                unpadded = len(score) - sum(score == -torch.finfo().max)
                num_token_masked = int(unpadded * rand_mask_prob)
                masked_indices = score.topk(num_token_masked, dim=-1).indices
                mask = torch.zeros_like(ids[i], dtype=torch.bool)
                mask[masked_indices] = True
                ids[i, masked_indices] = self.mask_token
                # keep indices which are not masked

                ids_to_keep[i, ~mask] = ids[i, ~mask]
            outputs = demask_fn.forward(
                src_input_id=src_input_id,  # target
                # self_cond_embed = self_cond_embed,
                tgt_input_id=ids,  # change to token id
                original_lens=original_lens,
                perturbation_id=perturbation_id,
                perturbation_embedding=perturbation_embedding,
                generate=True,
            )
            logits = outputs['logits']
            # Create a mask of already predicted tokens
            for sample in range(batch_size):
                unique_ids = torch.unique(ids_to_keep[sample])
                logits[sample, :, unique_ids] = -float('inf')
                
                # and also perturbed genes
                if self.perturbation_modeling == 'repression':
                    logits[sample, :, perturbation_id[sample]] = -float('inf')

            filtered_logits = top_k(logits, topk_filter_thres)
            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_token

            ids = torch.where(is_mask, pred_ids, ids)
            probs_without_temperature = logits.softmax(dim=-1)
            # avoid predicting the same token
            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')

            if not can_remask_prev_masked:
                scores = scores.masked_fill(~is_mask, -torch.finfo().max)
        count_outputs = self.decoder.forward(outputs['cls_embedding'])

        # set counts to 0 based on zero counts probability - To do: should we do this?
        # if 'zero_probs' in count_outputs.keys():
        #    bernoulli = torch.distributions.Bernoulli(probs=count_outputs['zero_probs'])
        #    zeros = bernoulli.sample() # sample from bernoulli probabilities
        #    count_outputs['count_lognorm'][zeros==0] = 0 # set 0 counts to 0            

        return count_outputs


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

    # # test dataloader
    # data_module = GeneformerDataModule(
    #     src_folder='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    #     'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_0h.dataset',
    #     tgt_folder='/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    #     'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_tokenised_degs_16h.dataset',
    #     max_len=334,
    # )
    # data_module.setup()
    # dataloader = data_module.train_dataloader()
    # # iterate through batches
    # src_train_iterator = iter(dataloader['src'])
    # tgt_train_iterator = iter(dataloader['tgt'])
    # src_batch = next(src_train_iterator)
    # tgt_batch = next(tgt_train_iterator)

    # (batch_size, seq_length)
    # position = PositionalEncoding(d_model, max_seq_length)
    # print(position(tgt_data).shape)
    # print(decoder(tgt_data, enc_output=src_data).shape)
    # out, label = transformer(
    #     src_batch['input_id'], src_batch['attention_mask'], tgt_batch['input_id']
    # )

    # src_data = torch.randint(20000,(10, 500))
    # src_attn_mask = torch.ones((10, 500))
    # src_attn_mask[:, 200:] = 0
    # tgt_data = torch.randint(20000,(10, n_tokens))
    # #pad
    # tgt_data[:, 100:] = 0
    # out, label = transformer(src_data, src_attn_mask, tgt_data)
    # set seed
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
        noise_schedule=cosine_schedule,
        tgt_input_id=label_tensor.to('cuda'),
        tgt_vocab_size=10,
        seq_length=12,
    )
