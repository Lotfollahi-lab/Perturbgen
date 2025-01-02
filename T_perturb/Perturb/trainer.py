import pickle
from typing import (
    Any,
    List,
    Literal,
)

import evaluate
import torch
import torch.ao.quantization

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CellGenTrainer
from T_perturb.Perturb.T_model import PerturberMasking
from T_perturb.src.utils import compute_rouge_score, return_perturbation_adata


class PerturberTrainer(CellGenTrainer):
    def __init__(
        self,
        generate: bool = False,
        sequence_length: int = 2048,
        temperature: float = 2.0,
        iterations: int = 18,
        mapping_dict_path: str | None = None,
        genes_to_perturb: List[int] | None = None,
        perturbation_mode: Literal['inference', 'generate'] | None = None,
        perturbation_sequence: Literal['src', 'tgt'] | None = None,
        perturbation_token: int | None = None,
        condition_dict: dict | None = None,
        batch_size: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transformer = PerturberMasking(
            tgt_vocab_size=kwargs['tgt_vocab_size'],
            d_model=kwargs['d_model'],
            num_heads=kwargs['num_heads'],
            num_layers=kwargs['num_layers'],
            d_ff=kwargs['d_ff'],
            max_seq_length=kwargs['max_seq_length'],
            dropout=kwargs['dropout'],
            pred_tps=kwargs['pred_tps'],
            n_total_tps=kwargs['n_total_tps'],
            encoder=kwargs['encoder'],
            mask_scheduler=kwargs['mask_scheduler'],
            pos_encoding_mode=kwargs['pos_encoding_mode'],
            return_attn=kwargs['return_attn'],
        )
        self.perturbation_mode = perturbation_mode
        if perturbation_mode is not None and genes_to_perturb is not None:
            self.perturbation_sequence = perturbation_sequence
            self.genes_to_perturb = genes_to_perturb
            if mapping_dict_path is not None:
                with open(
                    mapping_dict_path,
                    'rb',
                ) as f:
                    tokenid_to_gene = pickle.load(f)
                gene_to_token_id = {v: k for k, v in tokenid_to_gene.items()}
                tokens_to_perturb = [
                    gene_to_token_id[gene] for gene in self.genes_to_perturb
                ]
                self.tokens_to_perturb = torch.tensor(
                    tokens_to_perturb, dtype=torch.long
                )
            else:
                self.tokens_to_perturb = torch.tensor(
                    genes_to_perturb, dtype=torch.long
                )
            self.perturbation_token = torch.tensor(
                perturbation_token, dtype=torch.long, device=self.device
            )
            print(
                f'Start perturbation ...\n'
                f'- Perturbation mode: {self.perturbation_mode}\n'
                f'- Perturbation sequence: {self.perturbation_sequence}\n'
                f'- Perturbing genes: {genes_to_perturb}\n'
                f'- Replace with token: {perturbation_token}\n'
            )

        self.generate = generate
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.iterations = iterations

        self.test_dict['cls_cosine_similarity'] = []
        self.test_dict['mean_cosine_similarity'] = []
        self.test_dict['delta_probs'] = []

        self.rouge = evaluate.load('rouge')
        self.rouge_seq_len_list = [25, 100, kwargs['max_seq_length']]
        for seq_len in self.rouge_seq_len_list:
            self.test_dict[f'rouge1_{seq_len}'] = []
        self.max_seq_length = kwargs['max_seq_length']
        self.encoder = kwargs['encoder']
        self.pos_encoding_mode = kwargs['pos_encoding_mode']
        self.mask_scheduler = kwargs['mask_scheduler']

        self.condition_dict = condition_dict
        if self.condition_dict is not None:
            self.condition_ids = torch.zeros(
                batch_size,
                len(self.condition_dict),
                dtype=torch.long,
            )
            print(f'condition_ids: {self.condition_ids.shape}')

    # def quantize_model(self, model):
    #     return torch.ao.quantization.quantize_dynamic(
    #         model,
    #         {torch.nn.Linear},
    #         dtype=torch.qint8,
    #     )

    def forward(
        self,
        batch: Any,
        perturbation: bool = False,
        condition: bool = False,
    ):
        if perturbation:
            self.tokens_to_perturb = self.tokens_to_perturb.to(self.device)
            self.perturbation_token = self.perturbation_token.to(self.device)
        tgt_input_id_dict = {}
        for i in self.pred_tps:
            # create metadata tokens for cfg
            if condition and self.condition_dict is not None:
                for j, condition in enumerate(self.condition_dict.keys()):
                    condition_ids = batch[f'{condition}_t{i}']
                    condition_tokens = [
                        self.condition_dict[condition][id] for id in condition_ids
                    ]

                    self.condition_ids[:, j] = torch.tensor(
                        condition_tokens, dtype=torch.long
                    )

            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            if perturbation:
                if (
                    self.perturbation_sequence is not None
                    and 'tgt' in self.perturbation_sequence
                ):
                    perturbed_tgt = tgt_input_id_.clone()
                    mask = torch.isin(tgt_input_id_, self.tokens_to_perturb)

                    perturbed_tgt[mask] = self.perturbation_token
                    tgt_input_id_dict[f'tgt_input_ids_t{i}'] = perturbed_tgt
                else:
                    tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
            else:
                tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if perturbation:
            if (
                self.perturbation_sequence is not None
                and 'src' in self.perturbation_sequence
            ):
                perturbed_src = batch['src_input_ids'].clone()
                mask = torch.isin(batch['src_input_ids'], self.tokens_to_perturb)
                perturbed_src[mask] = self.perturbation_token

            else:
                perturbed_src = batch['src_input_ids']
        else:
            perturbed_src = batch['src_input_ids']

        if self.perturbation_mode is None:
            outputs = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=self.return_embeddings,
            )
        elif self.perturbation_mode == 'inference':
            outputs = self.transformer(
                src_input_id=perturbed_src,
                tgt_input_id_dict=tgt_input_id_dict,
                not_masked=True,
            )
        else:
            outputs = None

        return outputs, perturbed_src, tgt_input_id_dict

    def training_step(self, batch, *args, **kwargs):
        outputs, _, _ = self.forward(batch, perturbation=False, condition=True)
        for t in outputs.keys():
            dec_logits = outputs[t]['dec_logits']
            labels = outputs[t]['labels']
            with torch.no_grad():
                perp = self.perplexity(dec_logits, labels)
                self.log(
                    'train/perplexity',
                    perp,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch['tgt_input_ids_t1'].shape[0],
                    rank_zero_only=True,
                    sync_dist=True,
                )
            dec_logits = dec_logits.contiguous().view(-1, dec_logits.size(-1))
            labels = labels.contiguous().view(-1)
            masking_loss = self.masking_loss(dec_logits, labels)
            self.log(
                'train/masking_loss',
                masking_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['tgt_input_ids_t1'].shape[0],
                rank_zero_only=True,
                sync_dist=True,
            )
            return masking_loss

    def test_step(self, batch, *args, **kwargs):
        if self.perturbation_mode == 'inference':
            true_outputs, _, _ = self.forward(batch, perturbation=False)
            perturbed_outputs, _, _ = self.forward(batch, perturbation=True)

        elif self.perturbation_mode == 'generate':
            # print(self.transformer)
            # self.transformer = self.quantize_model(self.transformer)
            (
                _,
                pert_src_input_ids,
                tgt_input_id_dict,
            ) = self.forward(batch, perturbation=True)
            decoder_kwargs = {
                'tgt_input_id_dict': tgt_input_id_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
            }

            true_outputs, true_ids_dict = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                genes_to_perturb=self.tokens_to_perturb,
                **decoder_kwargs,
            )
            perturbed_outputs, perturbed_ids_dict = self.transformer.generate(
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
                f'Invalid perturbation mode: {self.perturbation_mode}:'
                f'Choose between "inference" or "generate"'
            )
        for t in self.pred_tps:
            true_cls = true_outputs[t]['dec_embedding'][:, 0, :]
            true_mean_cls = true_outputs[t]['mean_embedding']

            true_logits = true_outputs[t]['dec_logits']
            true_probs = torch.softmax(true_logits, dim=-1)
            true_probs = true_probs.sum(dim=1)

            perturbed_cls = perturbed_outputs[t]['dec_embedding'][:, 0, :]
            perturbed_mean_cls = perturbed_outputs[t]['mean_embedding']
            perturbed_logits = perturbed_outputs[t]['dec_logits']

            perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
            perturbed_probs = perturbed_probs.sum(dim=1)
            delta_probs = perturbed_probs - true_probs
            delta_cls_cos_sim = cosine_similarity(
                perturbed_cls,
                true_cls,
            )
            delta_mean_cos_sim = cosine_similarity(
                perturbed_mean_cls,
                true_mean_cls,
            )
            delta_cls_cos_sim = delta_cls_cos_sim.detach().cpu().to(torch.float16)
            delta_mean_cos_sim = delta_mean_cos_sim.detach().cpu().to(torch.float16)
            cls_embeddings = true_cls.detach().cpu().to(torch.float16)
            delta_probs = delta_probs.detach().cpu().to(torch.float16)
            self.test_dict['cls_cosine_similarity'].append(delta_cls_cos_sim)
            self.test_dict['mean_cosine_similarity'].append(delta_mean_cos_sim)
            self.test_dict['cls_embeddings'].append(cls_embeddings)
            self.test_dict['delta_probs'].append(delta_probs)

            # return obs_key
            self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{t}'])
            if len(self.var_list) > 0:
                for var in self.var_list:
                    self.test_dict[var].append(batch[f'{var}_t{t}'])

    def on_test_epoch_end(self):
        obs_key = self.var_list if len(self.var_list) > 0 else []
        obs_key.extend(['cell_idx'])
        return_perturbation_adata(
            test_dict=self.test_dict,
            obs_key=obs_key,
            output_dir=self.output_dir,
            file_name=(
                f'{self.date}_m{self.perturbation_mode}_adata'
                f'_g{self.genes_to_perturb}'
                f'_s{self.perturbation_sequence}'
                f'_t{self.perturbation_token}.h5ad'
            ),
            mode=self.perturbation_mode,
        )
