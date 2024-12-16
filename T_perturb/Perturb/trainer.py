from typing import Any, List

import torch

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CellGenTrainer, CountDecoderTrainer
from T_perturb.Perturb.T_model import PerturberGeneration
from T_perturb.src.utils import return_perturbation_adata  # WarmupScheduler,;


class PerturberInferenceTrainer(CellGenTrainer):
    def __init__(
        self,
        genes_to_perturb: List[int] | None = None,
        perturbation_token: int | None = 0,
        cell_type_to_perturb: str | None = None,
        perturbation_mode: List[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        print('test kwargs', kwargs)

        if perturbation_mode is not None:
            self.perturbation_mode = perturbation_mode
            self.genes_to_perturb = torch.tensor(genes_to_perturb, dtype=torch.long)
            self.perturbation_token = torch.tensor(perturbation_token, dtype=torch.long)
        else:
            self.perturbation_mode = []
        self.test_dict['cls_cosine_similarity'] = []
        self.test_dict['mean_cosine_similarity'] = []

    def forward(
        self,
        batch: Any,
        perturbation: bool = False,
    ):
        self.tgt_input_id_dict = {}
        for i in self.pred_tps:
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            if len(self.perturbation_mode) > 0 and perturbation:
                if 'tgt' in self.perturbation_mode:
                    print('perturbating tgt')
                    perturbed_tgt = tgt_input_id_.clone()
                    mask = torch.isin(tgt_input_id_, self.genes_to_perturb)
                    perturbed_tgt[mask] = self.perturbation_token
                self.tgt_input_id_dict[f'tgt_input_ids_t{i}'] = perturbed_tgt
            else:
                self.tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
        if len(self.perturbation_mode) > 0 and perturbation:
            if 'src' in self.perturbation_mode:
                print('perturbating src')
                perturbed_src = batch['src_input_ids'].clone()
                mask = torch.isin(batch['src_input_ids'], self.genes_to_perturb)
                perturbed_src[mask] = self.perturbation_token
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_dict=self.tgt_input_id_dict,
            not_masked=self.return_embeddings,
            context_mode=self.context_mode,
        )
        return outputs

    def test_step(self, batch, *args, **kwargs):
        if self.return_embeddings:
            true_outputs = self.forward(batch, perturbation=False)

            perturbed_outputs = self.forward(batch, perturbation=True)
            print('true', true_outputs.keys())
            print('true', true_outputs)
            print('perturbed', perturbed_outputs.keys())

            for t in self.pred_tps:
                print('t', t)
                print('true', true_outputs['dec_embedding'][t].shape)
                true_cls = true_outputs['dec_embedding'][t][:, 0, :]
                true_mean_cls = true_outputs['mean_embedding'][t]
                perturbed_cls = perturbed_outputs['dec_embedding'][t][:, 0, :]
                perturbed_mean_cls = perturbed_outputs['mean_embedding'][t]

                if len(self.perturbation_mode) > 0:
                    cls_cos_sim = cosine_similarity(
                        perturbed_cls,
                        true_cls,
                    )
                    print('cls_cos_sim', cls_cos_sim)
                    mean_agg_cos_sim = cosine_similarity(
                        perturbed_mean_cls,
                        true_mean_cls,
                    )
                    print('mean_agg_cos_sim', mean_agg_cos_sim)
                    self.test_dict['cls_cosine_similarity'].append(cls_cos_sim)
                    self.test_dict['mean_cosine_similarity'].append(mean_agg_cos_sim)

    def on_test_epoch_end(self):
        pass


class PerturberGenerationTrainer(CountDecoderTrainer):
    def __init__(
        self,
        genes_to_perturb: List[int] | None = None,
        perturbation_token: int | None = 0,
        cell_type_to_perturb: str | None = None,
        perturbation_mode: List[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        print('test kwargs', kwargs)

        self.decoder = PerturberGeneration(
            pretrained_model=self.pretrained_model,
            loss_mode=kwargs['loss_mode'],
            d_model=kwargs['d_model'],
            dropout=kwargs['dropout'],
            pred_tps=kwargs['pred_tps'],
            context_tps=kwargs['context_tps'],
            context_mode=kwargs['context_mode'],
            n_genes=kwargs['n_genes'],
        )
        if perturbation_mode is not None:
            self.perturbation_mode = perturbation_mode
            self.genes_to_perturb = torch.tensor(genes_to_perturb, dtype=torch.long)
            self.perturbation_token = torch.tensor(perturbation_token, dtype=torch.long)
        else:
            self.perturbation_mode = []
        self.test_dict['cls_cosine_similarity'] = []
        self.test_dict['mean_cosine_similarity'] = []

    def test_step(self, batch, *args, **kwargs):
        tgt_input_id_dict = {}
        for i in self.total_tps:
            print(i)
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
            if len(self.perturbation_mode) > 0:
                if 'tgt' in self.perturbation_mode:
                    print('perturbating tgt')
                    perturbed_tgt = batch[f'tgt_input_ids_t{i}'].clone()
                    mask = torch.isin(
                        batch[f'tgt_input_ids_t{i}'], self.genes_to_perturb
                    )
                    perturbed_tgt[mask] = self.perturbation_token
        if len(self.perturbation_mode) > 0:
            if 'src' in self.perturbation_mode:
                print('perturbating src')
                perturbed_src = batch['src_input_ids'].clone()
                mask = torch.isin(batch['src_input_ids'], self.genes_to_perturb)
                perturbed_src[mask] = self.perturbation_token
        if self.generate:
            decoder_kwargs = {
                'tgt_input_id_dict': tgt_input_id_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
                'sequence_length': self.sequence_length,
            }
            print('perturbation', self.perturbation_mode)
            if len(self.perturbation_mode) > 0:
                true_outputs, true_ids_dict = self.decoder.generate(
                    src_input_id=batch['src_input_ids'],
                    genes_to_perturb=self.genes_to_perturb,
                    **decoder_kwargs,
                )
                perturbed_outputs, perturbed_ids_dict = self.decoder.generate(
                    src_input_id=perturbed_src,
                    **decoder_kwargs,
                )

            else:
                true_outputs, true_ids_dict = self.decoder.generate(
                    src_input_id=batch['src_input_ids'],
                    **decoder_kwargs,
                )

            for t, time_step in enumerate(true_ids_dict.keys()):
                if len(self.perturbation_mode) > 0:
                    pred_ids = perturbed_ids_dict[time_step].detach().cpu().numpy()
                    tgt_ids = true_ids_dict[time_step].detach().cpu().numpy()
                    # compute cosine similarity between perturbed and true
                    # t = i + 1
                    print(perturbed_outputs[f'cls_embedding_t{t}'].shape)
                    cls_cos_sim = cosine_similarity(
                        perturbed_outputs[f'cls_embedding_t{t}'],
                        true_outputs[f'cls_embedding_t{t}'],
                    )
                    mean_agg_cos_sim = cosine_similarity(
                        perturbed_outputs[f'mean_embedding_t{t}'],
                        true_outputs[f'mean_embedding_t{t}'],
                    )
                    self.test_dict['cls_cosine_similarity'].append(cls_cos_sim)
                    self.test_dict['mean_cosine_similarity'].append(mean_agg_cos_sim)

                else:
                    pred_ids = true_ids_dict[time_step].detach().cpu().numpy()
                    tgt_ids = batch[time_step].detach().cpu().numpy()
                if self.return_rouge_score:
                    test_dict = self.compute_rouge_score(
                        pred_ids=pred_ids,
                        tgt_ids=tgt_ids,
                        rouge_len_list=self.rouge_seq_len_list,
                        max_seq_length=self.max_seq_length,
                        test_dict=self.test_dict,
                    )
                    self.test_dict = test_dict
                    # self.log(
                    #     'test/rouge1',
                    #     rouge_score['rouge1'],
                    #     on_step=False,
                    #     on_epoch=True,
                    #     prog_bar=True,
                    #     logger=True,
                    #     rank_zero_only=True,
                    #     sync_dist=True,
                    #     batch_size=batch['src_input_ids'].shape[0],
                    # )
            for time_step in self.pred_tps:
                self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{time_step}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])
                cls_embeddings = (
                    true_outputs[f'cls_embedding_t{time_step}'].detach().cpu()
                )
                self.test_dict['cls_embeddings'].append(cls_embeddings)

    def on_test_epoch_end(self):
        if self.generate:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['cell_idx'])
            return_perturbation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_generate_adata_'
                    f't{self.pred_tps}_{self.encoder}_s{self.seed}_'
                    f'l{self.loss_mode}_n{self.n_samples}'
                    f'_p{self.pos_encoding_mode}_'
                    f'm{self.mask_scheduler}_s{self.sequence_length}'
                ),
            )
